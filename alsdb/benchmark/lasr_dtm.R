# SPDX-License-Identifier: EUPL-1.2
# Independent DTM reference with lasR (r-lidar, GPL-3), run as a separate
# process so nothing GPL is linked into alsdb.
#
# From raw classified LAZ, entirely inside lasR: TIN of the ground and water
# returns (classes 2, 9, as alsdb's GROUND_CLASSES), rasterised at 1 m. Each
# tile is read with a 50 m buffer from its neighbours, as lasr_chm.R, so tile
# edges are interpolated, not extrapolated.
#
# Normally run through alsdb.benchmark.lasr.run_dtm() or
# `pixi run -e lasr python -m alsdb.benchmark lasr --dtm <laz_dir> <out_dir>`.
# Direct use:
#   Rscript lasr_dtm.R <laz_dir> <out_dir> [--ncores=8] [--tiles=N]
# Writes <out_dir>/dtm/<tile>.tif; skips tiles already written.

suppressPackageStartupMessages({
  library(lasR)
})

args <- commandArgs(trailingOnly = TRUE)
pos <- args[!startsWith(args, "--")]
if (length(pos) != 2) stop("usage: lasr_dtm.R <laz_dir> <out_dir> [--ncores=8] [--tiles=N]")
opt <- function(name, default) {
  hit <- grep(paste0("^--", name, "="), args, value = TRUE)
  if (length(hit)) sub(paste0("^--", name, "="), "", hit[1]) else default
}
laz_dir <- pos[1]
out_dir <- pos[2]
ncores <- as.integer(opt("ncores", 8))
n_tiles <- as.integer(opt("tiles", NA))
RES <- 1
BUFFER <- 50
GROUND <- "Classification %in% 2 9"

files <- sort(list.files(laz_dir, pattern = "(?i)\\.la[sz]$", full.names = TRUE))
if (!length(files)) stop("no .las/.laz files in ", laz_dir)
stem <- tools::file_path_sans_ext(basename(files))
dir.create(file.path(out_dir, "dtm"), FALSE, TRUE)
todo <- !file.exists(file.path(out_dir, "dtm", paste0(stem, ".tif")))
selected <- if (is.na(n_tiles)) rep(TRUE, length(files)) else seq_along(files) <= n_tiles
targets <- files[todo & selected]
message(sprintf("%d tiles, %d to process (%d cores)", length(files), length(targets), ncores))
if (!length(targets)) quit(save = "no")

header_bbox <- function(f) {
  con <- file(f, "rb"); h <- readBin(con, "raw", 227); close(con)
  d <- readBin(h[180:211], "double", n = 4, size = 8, endian = "little")
  c(xmin = d[2], ymin = d[4], xmax = d[1], ymax = d[3])
}
bbox <- t(vapply(files, header_bbox, numeric(4)))

process_tile <- function(f) {
  # Fixed 1 mm scale on read: see lasr_chm.R (PNOA header scale overflow).
  tri <- triangulate(filter = GROUND)
  p <- reader(filter = "-rescale 0.001 0.001 0.001 -auto_reoffset") + tri +
    lasR::rasterize(RES, tri, ofile = file.path(out_dir, "dtm", "*.tif"))
  b <- bbox[f, ]
  near <- bbox[, "xmin"] <= b[["xmax"]] + BUFFER & bbox[, "xmax"] >= b[["xmin"]] - BUFFER &
    bbox[, "ymin"] <= b[["ymax"]] + BUFFER & bbox[, "ymax"] >= b[["ymin"]] - BUFFER
  on <- c(f, setdiff(files[near], f))
  invisible(exec(p, on = on, with = list(ncores = sequential(), progress = FALSE,
                                         buffer = BUFFER, noprocess = on != f, noread = TRUE)))
  basename(f)
}

# Spatial indexes first, then fresh PSOCK workers (see lasr_chm.R for why).
need <- files[!file.exists(sub("[.]la[sz]$", ".lax", files, ignore.case = TRUE))]
if (length(need)) {
  message(sprintf("writing %d .lax spatial indexes next to the input files", length(need)))
  invisible(exec(write_lax(), on = need, with = list(ncores = concurrent_files(ncores), progress = FALSE)))
}
cl <- parallel::makePSOCKcluster(min(ncores, length(targets)))
invisible(parallel::clusterEvalQ(cl, suppressPackageStartupMessages(library(lasR))))
parallel::clusterExport(cl, c("process_tile", "bbox", "files", "BUFFER", "RES", "GROUND", "out_dir"))
results <- parallel::parLapplyLB(cl, targets, function(f) tryCatch(process_tile(f), error = function(e) e))
parallel::stopCluster(cl)
ok <- vapply(results, is.character, logical(1))
for (i in which(!ok)) message("FAILED ", basename(targets[i]), ": ", conditionMessage(results[[i]]))
message(sprintf("done: %d of %d tiles", sum(ok), length(targets)))
if (!all(ok)) quit(save = "no", status = 1)
