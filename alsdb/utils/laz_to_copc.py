
from pathlib import Path
import subprocess
import time


class LAZToCOPCConverter:
    """Convert a directory of LAZ files into COPC files."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.input_dir.is_dir():
            raise NotADirectoryError(
                f"Input directory does not exist: {self.input_dir}"
            )

    def convert_file(self, laz_file: Path) -> bool:
        """Convert a single LAZ file into COPC."""

        copc_file = self.output_dir / f"{laz_file.stem}.copc.laz"

        if copc_file.exists():
            print(f"Skipping {laz_file.name}")
            return True

        temp_file = self.output_dir / f"{laz_file.stem}.tmp.copc.laz"

        print(f"Converting {laz_file.name}")

        try:
            subprocess.run(
                [
                    "pdal",
                    "translate",
                    str(laz_file),
                    str(temp_file),
                ],
                check=True,
            )

            temp_file.replace(copc_file)
            return True

        except subprocess.CalledProcessError as e:
            print(f"ERROR converting {laz_file.name}: {e}")

            if temp_file.exists():
                temp_file.unlink()

            return False

    def convert(self) -> None:
        """Convert all LAZ files in the input directory."""

        laz_files = sorted(self.input_dir.glob("*.laz"))

        print(f"Found {len(laz_files)} LAZ files")

        start = time.time()
        successful = 0
        failed = 0

        for i, laz_file in enumerate(laz_files, 1):
            print(f"[{i}/{len(laz_files)}] ", end="")

            if self.convert_file(laz_file):
                successful += 1
            else:
                failed += 1

        elapsed = (time.time() - start) / 60

        print("\nConversion finished")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total time: {elapsed:.2f} minutes")


if __name__ == "__main__":

    converter = LAZToCOPCConverter(
        input_dir="./testdata/barcelona/laz",
        output_dir="./testdata/barcelona/copc",
    )

    converter.convert()
