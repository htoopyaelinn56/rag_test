from pathlib import Path

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption

from chunker import process_chunk


def main():
    input_doc_path = Path('project_proposal.pdf')

    # Pre-check: provide a clear error if the file is missing or name differs
    if not input_doc_path.exists():
        raise FileNotFoundError(
            f"Input PDF not found at: {input_doc_path}\n"
            "Tips: check the exact filename (including spaces/parentheses), and run the script from the project folder."
        )

    accelerator_options = AcceleratorOptions(
        num_threads=8, device=AcceleratorDevice.MPS
    )

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False
    pipeline_options.table_structure_options.do_cell_matching = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    # Enable the profiling to measure the time spent
    settings.debug.profile_pipeline_timings = True

    print("Beginning conversion...")
    # Convert the document
    conversion_result = converter.convert(input_doc_path)
    doc = conversion_result.document

    # List with total time per document
    doc_conversion_secs = conversion_result.timings["pipeline_total"].times
    print(f"Conversion secs: {doc_conversion_secs}")

    process_chunk(doc)

if __name__ == "__main__":
    main()