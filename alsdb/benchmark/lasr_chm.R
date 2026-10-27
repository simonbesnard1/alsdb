# SPDX-License-Identifier: EUPL-1.2
# Independent open-source CHM reference with lasR (r-lidar, GPL-3), run as a
# separate process so nothing GPL is linked into alsdb.
#
# From raw classified LAZ, entirely inside lasR:
#   1. drop noise (classes 7, 18)
#   2. normalize: TIN of ground + water (classes 2, 9), Z -> height
#   3. spikefree: Khosravipour 2016 on all returns of classes 2-5 and 9
#   4. pitfree:   Khosravipour 2014 / lidR pitfree(), as alsdb algorithm 3:
#      highest first return per 1 m cell; base layer = all surface returns
#      (ground included), untrimmed; layers >= 2, 5, 10, 15, 20 m trimmed at
#      triangle edges > EDGE; per-cell max of the layers.
# Same class selection and parameters as alsdb surface algorithm 3, so
# differences are implementation, not settings. --freeze/--edge default to
# "auto", alsdb's rule computed per LAZ tile from its surface-class first
# returns (classes 2, 9 and --veg; PNOA overlap-strip returns, class 0 with
# the Overlap flag, are excluded like in alsdb):
#   freeze = 3 x spacing of those first returns
#   edge   = 3 x spacing after keeping one per 1 m cell
#   spacing = sqrt(occupied 5 m cell area / count)
# alsdb evaluates the same rule per buffered 1 km processing tile, so values
# differ slightly. Barcelona 400-4604: freeze 4.4, edge 4.5 m in both. With
# equal values, pitfree matched alsdb on 99.3% of cells (<1 cm) and
# spikefree on 89% (mean +0.09 m) on that tile. Values used: params.csv.
#
# Normally run through alsdb.benchmark.lasr.run() or
# `pixi run -e lasr python -m alsdb.benchmark lasr <laz_dir> <out_dir>`.
# Direct use:
#   Rscript lasr_chm.R <laz_dir> <out_dir> [--ncores=8] [--tiles=N]
#       [--freeze=auto] [--edge=auto] [--veg=3,4,5] [--normalized]
# --veg: vegetation classes (alsdb veg_classes); e.g. --veg=20 for DE-Hai.
# --freeze=0 selects lasR's locally adaptive spike-free (Fischer et al. 2024).
# --normalized: input Z is already height above ground (e.g. LAStools
#   HAG_classfixed/), so step 2 is skipped. Prefer the raw LAZ (lasR then
#   normalizes itself).
# All inputs are read at a fixed 1 mm scale (-rescale, -auto_reoffset): PNOA's tiny header
# scale factors otherwise overflow when neighbour headers are merged.
# Raw LAZ without .lax spatial indexes works but buffered reads are slower;
# lasR's write_lax() can create them (it writes next to the input files).
# Writes <out_dir>/spikefree/<tile>.tif and <out_dir>/pitfree/<tile>.tif.
# Skips tiles whose two outputs already exist (rerun to resume).

suppressPackageStartupMessages({
  library(lasR)
  library(terra)
})

args <- commandArgs(trailingOnly = TRUE)
pos <- args[!startsWith(args, "--")]
if (length(pos) != 2) stop("usage: lasr_chm.R <laz_dir> <out_dir> [--ncores=8] [--tiles=N] [--freeze=auto] [--edge=auto] [--veg=3,4,5] [--normalized]")
opt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^--", name, "="), "", hit[1]) else default
}
laz_dir <- pos[1]
out_dir <- pos[2]
ncores <- as.integer(opt("ncores", 8))
n_tiles <- as.integer(opt("tiles", NA))
FREEZE <- opt("freeze", "auto")
EDGE <- opt("edge", "auto")
NORMALIZED <- "--normalized" %in% args
veg <- as.integer(strsplit(opt("veg", "3,4,5"), ",")[[1]])
RES <- 1
THRESHOLDS <- c(2, 5, 10, 15, 20)
SURFACE <- paste("Classification %in%", paste(sort(unique(c(2L, 9L, veg))), collapse = " "))
VEG <- paste("Classification %in%", paste(veg, collapse = " "))

files <- sort(list.files(laz_dir, pattern = "(?i)\\.la[sz]$", full.names = TRUE))
if (!length(files)) stop("no .las/.laz files in ", laz_dir)
stem <- tools::file_path_sans_ext(basename(files))
todo <- !(file.exists(file.path(out_dir, "spikefree", paste0(stem, ".tif"))) &
          file.exists(file.path(out_dir, "pitfree", paste0(stem, ".tif"))))
# --tiles N = the first N tiles (as the LAStools runner), finished ones skipped,
# so repeating a pilot does not move on to new tiles.
selected <- if (is.na(n_tiles)) rep(TRUE, length(files)) else seq_along(files) <= n_tiles
targets <- files[todo & selected]
message(sprintf("%d tiles, %d to process (freeze %s, edge %s, veg %s, %d cores, %s input)",
                length(files), length(targets), FREEZE, EDGE, paste(veg, collapse = ","),
                ncores, if (NORMALIZED) "normalized" else "raw"))
if (!length(targets)) quit(save = "no")

layer_dir <- file.path(out_dir, "pitfree_layers")
for (d in c("spikefree", "pitfree", layer_dir)) dir.create(file.path(out_dir, basename(d)), FALSE, TRUE)

# alsdb's auto rule, per tile, from a 1 m count raster of surface first returns:
# sum = returns, non-empty 1 m cells = one-highest-per-cell count, non-empty
# 5 m cells = occupied area.
auto_params <- function(f) {
  # Named per tile and process: forked workers share R's random state, so
  # tempfile() returned the same name in several workers at once.
  tmp <- file.path(tempdir(), sprintf("count_%s_%d.tif",
                                      tools::file_path_sans_ext(basename(f)), Sys.getpid()))
  exec(lasR::rasterize(RES, "count", filter = c(keep_first(), SURFACE), ofile = tmp), on = f,
       with = list(ncores = sequential(), progress = FALSE, noread = TRUE))
  r <- rast(tmp)
  v <- values(r, mat = FALSE)
  n_raw <- sum(v, na.rm = TRUE)
  n_thin <- sum(!is.na(v) & v > 0)
  occ5 <- aggregate(r, 5, fun = "sum", na.rm = TRUE)
  area <- sum(values(occ5, mat = FALSE) > 0, na.rm = TRUE) * 25
  unlink(tmp)
  c(freeze = 3 * sqrt(area / n_raw), edge = 3 * sqrt(area / n_thin))
}
make_pipeline <- function(freeze, edge) {
  # PNOA headers use scale factors down to ~4e-08 m. Merging neighbour tiles
  # for the buffer makes LASlib enlarge them 10x, after which Z above ~894 m
  # overflows the 32-bit integers and spikefree fails ("Point are not
  # inserted in Z order"). A fixed 1 mm scale, set on read, avoids that;
  # -auto_reoffset keeps coordinates in range for files with zero offsets.
  p <- reader(filter = "-rescale 0.001 0.001 0.001 -auto_reoffset") + delete_points(keep_noise())
  if (!NORMALIZED) p <- p + normalize()
  # Points beyond the ground TIN (e.g. at the survey edge) get no valid ground
  # from lasR's normalize and come out at around -1e6 m. Drop them so those
  # cells are NoData, as alsdb leaves cells without ground support.
  p <- p + delete_points("Z < -5")
  p <- p +
    spikefree(RES, freeze, 0.5, filter = SURFACE, ofile = file.path(out_dir, "spikefree", "*.tif")) +
    # Everything after this sees the thinned cloud: highest first return per cell.
    sampling_pixel(RES, filter = c(keep_first(), SURFACE), method = "max")
  base <- triangulate(0, filter = c(keep_first(), SURFACE))
  p <- p + base + lasR::rasterize(RES, base, ofile = file.path(layer_dir, "*_l00.tif"))
  for (th in THRESHOLDS) {
    tri <- triangulate(edge, filter = c(keep_first(), VEG, paste("Z >=", th)))
    p <- p + tri + lasR::rasterize(RES, tri, ofile = file.path(layer_dir, sprintf("*_l%02d.tif", th)))
  }
  p
}

# Header bounding boxes, read once: each tile gets only the neighbours its
# 50 m buffer touches instead of all files (lasR reads every header in `on`).
header_bbox <- function(f) {
  con <- file(f, "rb"); h <- readBin(con, "raw", 227); close(con)
  d <- readBin(h[180:211], "double", n = 4, size = 8, endian = "little")
  c(xmin = d[2], ymin = d[4], xmax = d[1], ymax = d[3])  # stored max x, min x, max y, min y
}
bbox <- t(vapply(files, header_bbox, numeric(4)))
BUFFER <- 50

# One worker process per tile, each with its own auto parameters. (Grouping
# tiles by equal parameters ran the groups one after another, so tiles with
# different values never ran in parallel.)
process_tile <- function(f) {
  a <- if (FREEZE == "auto" || EDGE == "auto") auto_params(f) else c(freeze = NA, edge = NA)
  freeze <- round(if (FREEZE == "auto") a[["freeze"]] else as.numeric(FREEZE), 1)
  edge <- round(if (EDGE == "auto") a[["edge"]] else as.numeric(EDGE), 1)
  b <- bbox[f, ]
  near <- bbox[, "xmin"] <= b[["xmax"]] + BUFFER & bbox[, "xmax"] >= b[["xmin"]] - BUFFER &
    bbox[, "ymin"] <= b[["ymax"]] + BUFFER & bbox[, "ymax"] >= b[["ymin"]] - BUFFER
  on <- c(f, setdiff(files[near], f))
  invisible(exec(make_pipeline(freeze, edge), on = on,
                 with = list(ncores = sequential(), progress = FALSE, buffer = BUFFER,
                             noprocess = on != f, noread = TRUE)))
  s <- tools::file_path_sans_ext(basename(f))
  layers <- list.files(layer_dir, pattern = paste0("^", s, "_l\\d+\\.tif$"), full.names = TRUE)
  if (!length(layers)) stop("no pitfree layers for ", s)
  rs <- lapply(sort(layers), rast)
  # Sparse upper layers may cover a smaller extent; pad all to the union.
  e <- Reduce(terra::union, lapply(rs, ext))
  stack <- rast(lapply(rs, extend, y = e))
  writeRaster(max(stack, na.rm = TRUE), file.path(out_dir, "pitfree", paste0(s, ".tif")),
              overwrite = TRUE, datatype = "FLT4S")
  invisible(unlink(layers))
  data.frame(freeze = freeze, edge = edge, tile = basename(f))
}

# Build missing .lax spatial indexes once, before the workers start. lasR
# otherwise writes them on demand next to the input files, and parallel
# workers reading the same neighbour tile for their buffers raced on it:
# tiles were then processed wrongly, outputs mixed up and corrupted. With the
# indexes in place, 1 and 4 workers give identical pitfree and spike-free
# within lasR's own run-to-run variation (~0.1-0.3% of cells).
need <- files[!file.exists(sub("[.]la[sz]$", ".lax", files, ignore.case = TRUE))]
if (length(need)) {
  message(sprintf("writing %d .lax spatial indexes next to the input files", length(need)))
  invisible(exec(write_lax(), on = need, with = list(ncores = concurrent_files(ncores), progress = FALSE)))
}

# Fresh R processes, not forks: forked workers (mclapply) inherit the parent's
# library state (OpenMP, GDAL) and lasR in them intermittently wrote no output
# without raising an error. PSOCK workers start clean.
cl <- parallel::makePSOCKcluster(min(ncores, length(targets)))
invisible(parallel::clusterEvalQ(cl, suppressPackageStartupMessages({ library(lasR); library(terra) })))
parallel::clusterExport(cl, c("auto_params", "make_pipeline", "process_tile", "bbox", "files",
                              "BUFFER", "RES", "THRESHOLDS", "SURFACE", "VEG", "FREEZE", "EDGE",
                              "NORMALIZED", "out_dir", "layer_dir"))
results <- parallel::parLapplyLB(cl, targets, function(f) tryCatch(process_tile(f), error = function(e) e))
parallel::stopCluster(cl)
ok <- vapply(results, is.data.frame, logical(1))
if (any(ok)) {
  log_file <- file.path(out_dir, "params.csv")
  write.table(do.call(rbind, results[ok]), log_file, sep = ",", row.names = FALSE,
              append = file.exists(log_file), col.names = !file.exists(log_file))
}
for (i in which(!ok)) message("FAILED ", basename(targets[i]), ": ", conditionMessage(results[[i]]))
message(sprintf("done: %d of %d tiles", sum(ok), length(targets)))
if (!all(ok)) quit(save = "no", status = 1)
