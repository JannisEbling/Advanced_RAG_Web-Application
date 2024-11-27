import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from azure.ai.documentintelligence.models import AnalyzeResult, ContentFormat
from azure.core.credentials import AzureKeyCredential
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import fitz  # PyMuPDF
import base64
from pathlib import Path
from glob import glob

# Azure Document Intelligence credentials
endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
pdf_file = "data/test"


class DocumentIntelligenceClientWrapper:
    def __init__(self):
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        # Use absolute paths
        self.output_dir = Path.cwd() / "analysis_results"
        self.figures_dir = self.output_dir / "figures"

        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

        # Store formulas for later use
        self.formulas_data = []

    def analyze_documents(self, pdf_paths):
        """
        Analyze a list of PDF documents and extract paragraphs, tables, and images.

        Args:
            pdf_paths: List of paths to PDF files to analyze

        Returns:
            dict: Dictionary containing extracted data for each PDF:
                {
                    'file_path': {
                        'paragraphs': List of extracted paragraphs,
                        'tables': List of extracted table data,
                        'images': List of extracted image data
                    }
                }
        """
        results = {}
        try:
            for pdf_path in pdf_paths:
                # Load the PDF document
                pdf_document = fitz.open(pdf_path)

                # Read the document and encode it in base64
                with open(pdf_path, "rb") as f:
                    document_content = f.read()
                    base64_encoded = base64.b64encode(document_content).decode()

                poller = self.client.begin_analyze_document(
                    "prebuilt-layout",
                    {"base64Source": base64_encoded},
                    output_content_format=ContentFormat.MARKDOWN,
                    # features=[
                    #     DocumentAnalysisFeature.FORMULAS
                    # ],  # Enable figures and styles extraction
                )
                result: AnalyzeResult = poller.result()
                self.results = result

                # Print formula information if available
                self.formulas_data = []

                # Process tables if available
                if hasattr(result, "tables") and result.tables:
                    tables_data = self.process_tables(pdf_document, pdf_path)
                else:
                    tables_data = []

                # Process paragraphs with roles
                if hasattr(result, "paragraphs") and result.paragraphs:
                    paragraphs_data = self.process_paragraphs(pdf_document, pdf_path)
                else:
                    paragraphs_data = []

                # Process figures if available
                if hasattr(result, "figures") and result.figures:
                    figures_data = self.process_images(pdf_document, pdf_path)
                else:
                    figures_data = []

                # Close the PDF document
                pdf_document.close()

                doc_results = {
                    'paragraphs': paragraphs_data,
                    'tables': tables_data,
                    'images': figures_data
                }
                results[pdf_path] = doc_results

                print(f"Analysis complete for {pdf_path}!")
                print(f"Visualizations saved as PNG files in {self.output_dir}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            raise

        return results

    def process_paragraphs(self, pdf_document, pdf_path):
        print(f"\nFound {len(self.results.paragraphs)} paragraphs in the document")
        paragraphs_data = []

        # Create a mapping of spans to figure IDs
        figure_spans = {}

        # Create a mapping of spans to formula IDs
        formula_spans = {}
        for formula in self.formulas_data:
            for span in formula["spans"]:
                span_key = (span["offset"], span["length"])
                formula_spans[span_key] = formula["id"]

        for idx, paragraph in enumerate(self.results.paragraphs):
            # Find if this paragraph is related to any figure
            related_figures = set()
            if hasattr(paragraph, "spans"):
                for span in paragraph.spans:
                    span_key = (span.offset, span.length)
                    if span_key in figure_spans:
                        related_figures.add(figure_spans[span_key])

            # Find if this paragraph contains any formulas
            related_formulas = set()
            if hasattr(paragraph, "spans"):
                for span in paragraph.spans:
                    span_key = (span.offset, span.length)
                    if span_key in formula_spans:
                        related_formulas.add(formula_spans[span_key])

            paragraph_info = {
                "id": idx,
                "content": paragraph.content,
                "role": paragraph.role if hasattr(paragraph, "role") else "undefined",
                "related_figures": list(related_figures),
                "related_formulas": list(related_formulas),
                "spans": [
                    {"offset": span.offset, "length": span.length}
                    for span in paragraph.spans
                ]
                if hasattr(paragraph, "spans")
                else [],
                "locations": [],
            }

            # Process bounding regions if available
            if hasattr(paragraph, "bounding_regions") and paragraph.bounding_regions:
                for region in paragraph.bounding_regions:
                    location = {
                        "page": region.page_number,
                        "polygon": region.polygon,
                    }
                    paragraph_info["locations"].append(location)
                    print(f"\nParagraph #{idx}:")
                    print(f"- Role: {paragraph_info['role']}")
                    print(
                        f"- Content: {paragraph.content[:100]}..."
                    )  # Show first 100 chars
                    print(f"- Page: {region.page_number}")
                    print(f"- Coordinates: {region.polygon}")
                    if related_figures:
                        print(f"- Related to figures: {list(related_figures)}")
                    if related_formulas:
                        print(f"- Related to formulas: {list(related_formulas)}")

            paragraphs_data.append(paragraph_info)

        # Save paragraphs data to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.paragraphs_output_path = (
            self.output_dir / f"paragraphs_data_{timestamp}.json"
        )

        with open(self.paragraphs_output_path, "w", encoding="utf-8") as f:
            json.dump(paragraphs_data, f, indent=2, ensure_ascii=False)

        print(f"\nParagraphs data saved to {self.paragraphs_output_path}")
        return paragraphs_data

    def process_images(self, pdf_document, pdf_path):
        print(f"\nFound {len(self.results.figures)} figures in the document")
        figures_data = []

        for idx, figure in enumerate(self.results.figures):
            # Get paragraph ID from elements (e.g., '/paragraphs/50' -> 50)
            figure_name = f"figure_{idx}"  # default name
            if hasattr(figure, "caption") and figure.caption:
                caption = figure.caption  # Use attribute access
                if hasattr(caption, "elements") and caption.elements:
                    for element in caption.elements:
                        if isinstance(element, str) and element.startswith(
                            "/paragraphs/"
                        ):
                            para_idx = int(element.split("/")[-1])
                            if hasattr(self.results, "paragraphs") and para_idx < len(
                                self.results.paragraphs
                            ):
                                para = self.results.paragraphs[para_idx]
                                if hasattr(para, "content") and para.content.startswith(
                                    "Figure"
                                ):
                                    # Extract "Figure X.Y" from the content
                                    figure_name = para.content.split(" ", 2)[0:2]
                                    figure_name = "_".join(figure_name)
                                    break
            # Convert caption to dictionary if it exists
            caption_dict = None
            if hasattr(figure, "caption"):
                caption_dict = {
                    "content": figure.caption.content
                    if hasattr(figure.caption, "content")
                    else None,
                    "spans": [
                        {"offset": s.offset, "length": s.length}
                        for s in figure.caption.spans
                    ]
                    if hasattr(figure.caption, "spans")
                    else [],
                }

            figure_info = {
                "id": idx,
                "caption": caption_dict,
                "spans": [
                    {"offset": span.offset, "length": span.length}
                    for span in figure.spans
                ]
                if hasattr(figure, "spans")
                else [],
                "locations": [],
            }

            # Process bounding regions for each figure
            if hasattr(figure, "bounding_regions"):
                for region_idx, region in enumerate(figure.bounding_regions):
                    location = {"page": region.page_number, "polygon": region.polygon}
                    figure_info["locations"].append(location)
                    print(f"\nFigure #{idx}:")
                    print(f"- Page: {region.page_number}")
                    print(f"- Coordinates: {region.polygon}")

                    # Extract image using PyMuPDF
                    page = pdf_document[region.page_number - 1]  # 0-based index

                    try:
                        # Get page dimensions from MediaBox
                        mediabox = page.mediabox
                        page_width = mediabox.width
                        page_height = mediabox.height

                        # Convert polygon coordinates to rectangle
                        polygon = region.polygon
                        x_coords = polygon[::2]
                        y_coords = polygon[1::2]

                        # Convert from PDF points (72 points per inch) to page coordinates
                        points_per_inch = 72.0
                        x_coords = [x * points_per_inch for x in x_coords]
                        y_coords = [y * points_per_inch for y in y_coords]

                        # Create rectangle
                        rect = fitz.Rect(
                            min(x_coords), min(y_coords), max(x_coords), max(y_coords)
                        )

                        # Debug information
                        print(f"\nDebug information for Figure #{idx}:")
                        print(f"MediaBox dimensions: {page_width} x {page_height}")
                        print(f"Original coordinates - X: {polygon[::2]}")
                        print(f"Original coordinates - Y: {polygon[1::2]}")
                        print(f"Scaled coordinates - X: {x_coords}")
                        print(f"Scaled coordinates - Y: {y_coords}")
                        print(f"Final rectangle: {rect}")

                        # Check if rectangle is valid and within page bounds
                        if (
                            rect.is_valid
                            and rect.width > 0
                            and rect.height > 0
                            and rect.x0 >= 0
                            and rect.y0 >= 0
                            and rect.x1 <= page_width
                            and rect.y1 <= page_height
                        ):
                            # Extract image from the page with higher resolution
                            mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                            try:
                                pix = page.get_pixmap(matrix=mat, clip=rect)
                                image_path = os.path.join(
                                    str(self.figures_dir),
                                    f"{figure_name}.png",
                                )
                                pix.save(image_path)
                                figure_info["image_path"] = image_path
                                print(f"Successfully extracted figure to: {image_path}")
                            except Exception as e:
                                print(f"Error during pixmap creation: {str(e)}")
                                figure_info["image_path"] = None
                        else:
                            print(
                                f"Warning: Invalid rectangle dimensions or out of bounds for Figure #{idx}"
                            )
                            print(f"Page bounds: (0, 0, {page_width}, {page_height})")
                            print(
                                f"Rectangle: ({rect.x0}, {rect.y0}, {rect.x1}, {rect.y1})"
                            )
                            figure_info["image_path"] = None

                    except Exception as e:
                        print(f"Error extracting Figure #{idx}: {str(e)}")
                        figure_info["image_path"] = None

            figures_data.append(figure_info)

        # Save figures data to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        figures_output_path = self.output_dir / f"figures_data_{timestamp}.json"
        with open(figures_output_path, "w", encoding="utf-8") as f:
            json.dump(figures_data, f, indent=2)

        print(f"\nFigures data saved to {figures_output_path}")
        print(f"Extracted figures saved to {self.figures_dir}")

        return figures_data

    def process_tables(self, pdf_document, pdf_path):
        print(f"\nFound {len(self.results.tables)} tables in the document")
        tables_data = []

        for table_idx, table in enumerate(self.results.tables):
            print(
                f"Table #{table_idx} has {table.row_count} rows and {table.column_count} columns"
            )
            if table.bounding_regions:
                for region in table.bounding_regions:
                    print(
                        f"Table #{table_idx} location on page: {region.page_number} is {region.polygon}"
                    )

            # Analyze cells if available
            if hasattr(table, "cells") and table.cells:
                for cell in table.cells:
                    print(
                        f"...Cell[{cell.row_index}][{cell.column_index}] has text '{cell.content}'"
                    )
                    if cell.bounding_regions:
                        for region in cell.bounding_regions:
                            print(
                                f"...content on page {region.page_number} is within bounding polygon '{region.polygon}'"
                            )

            table_info = {
                "id": table_idx,
                "rows": table.row_count,
                "columns": table.column_count,
                "cells": [
                    {
                        "row": cell.row_index,
                        "column": cell.column_index,
                        "content": cell.content,
                    }
                    for cell in table.cells
                ],
                "locations": [
                    {"page": region.page_number, "polygon": region.polygon}
                    for region in table.bounding_regions
                ],
            }

            tables_data.append(table_info)

        return tables_data


if __name__ == "__main__":
    if not endpoint or not key:
        print(
            "Please set the AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables."
        )
    else:
        documentintelligence = DocumentIntelligenceClientWrapper()
        pdf_paths = glob(os.path.join(pdf_file, "*.pdf"))
        results = documentintelligence.analyze_documents(pdf_paths)
        print(results)
