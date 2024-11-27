from typing import List, Dict, Any, Optional, Tuple
import json
import re
import os
from dataclasses import dataclass, asdict
from collections import defaultdict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class DocumentMetadata:
    page_number: int
    section_heading: str
    subsection_heading: str
    page_header: str
    figure_references: List[str]
    formula_references: List[str]


class AzureResponseProcessor:
    """Class to process and extract information from Azure Form Recognizer responses."""

    def __init__(self, response: Any = None):
        self.response = response
        self.current_metadata = DocumentMetadata(
            page_number=0,
            section_heading="",
            subsection_heading="",
            page_header="",
            figure_references=[],
            formula_references=[],
        )
        self.formulas = []
        self.figures = []
        self.documents = []
        self.formula_buffer = []
        self._title_pages = set()  # Track pages with titles
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        self.subsection_header = ""  # Class variable to track current subsection
        self.section_header = ""  # Class variable to track currentsubsection

    def _group_paragraphs(self, paragraphs: List[Dict]) -> List[Document]:
        """Group paragraphs by metadata and create Documents."""
        # Group paragraphs by their metadata keys
        grouped = defaultdict(list)
        page_numbers = defaultdict(set)  # Track page numbers for each group
        metadata_map = defaultdict(dict)  # Store metadata for each group

        for para in paragraphs:
            if not para:  # Skip None or empty paragraphs
                continue

            meta = para.get("metadata", {})
            key = (
                meta.get("section_heading", ""),
                meta.get("subsection_heading", ""),
                meta.get("page_header", ""),
            )
            grouped[key].append(para["content"])
            metadata_map[key] = meta  # Store the metadata for this group

            # Track page numbers from metadata
            page_num = meta.get("page_number", 0)
            page_numbers[key].add(page_num)

        # Create documents from grouped paragraphs
        documents = []
        for key, contents in grouped.items():
            # Join contents with newlines
            text = "\n".join(contents)
            section, subsection, header = key
            original_meta = metadata_map[key]

            # Get minimum page number for this group
            min_page = min(page_numbers[key]) if page_numbers[key] else 0

            # Split text into chunks
            chunks = self.text_splitter.split_text(text)

            # Create Document for each chunk
            for chunk in chunks:
                # Process chunk for references
                figure_refs = set()  # Use sets to automatically remove duplicates
                formula_refs = set()

                # Check for formula references
                formula_matches = re.finditer(r"\(\d+\.\d+\)", chunk)
                for match in formula_matches:
                    formula_refs.add(match.group(0))

                # Check for figure references
                figure_matches = re.finditer(r"Figure \d+\.\d+", chunk)
                for match in figure_matches:
                    # Replace spaces with underscores in figure references
                    figure_ref = match.group(0).replace(" ", "_")
                    figure_refs.add(figure_ref)

                # Create metadata
                metadata = {
                    "section_heading": key[0],
                    "subsection_heading": key[1],  # Use original metadata
                    "page_header": key[2],
                    "page_number": min_page,
                    "figure_references": sorted(list(figure_refs)),
                    "formula_references": sorted(list(formula_refs)),
                }

                # Create Document
                doc = Document(page_content=chunk, metadata=metadata)
                documents.append(doc)

        return documents

    def process_paragraphs(self) -> Tuple[List[Document], List[Dict], List[Dict]]:
        """
        Process all paragraphs in the response.

        Returns:
            Tuple containing:
            - List of processed Documents
            - List of formulas
            - List of figures
        """
        if not self.response:
            return [], [], []

        processed_paragraphs = []

        for paragraph in self.response:
            result = self.process_paragraph(paragraph)

            if result:
                if self.formula_buffer:
                    # Check if previous short paragraphs form a formula
                    combined_content = " ".join(
                        [p["content"] for p in self.formula_buffer]
                    )
                    is_formula, formula_ref, clean_content = (
                        self._get_formula_reference(combined_content)
                    )
                    if is_formula:
                        formula_obj = {
                            "content": clean_content,
                            "formula_reference": formula_ref,
                            "locations": [
                                loc
                                for p in self.formula_buffer
                                for loc in p.get("locations", [])
                            ],
                        }
                        # if formula_obj not in self.formulas:
                        # self.formulas.append(formula_obj)
                    self.formula_buffer = []
                processed_paragraphs.append(result)
            elif len(paragraph.get("content", "")) < 15:
                self.formula_buffer.append(paragraph)
            else:
                self.formula_buffer = []

        # Group paragraphs and create Documents
        self.documents = self._group_paragraphs(processed_paragraphs)

        return self.documents, self.formulas, self.figures

    def set_response(self, response: Any) -> None:
        """Set a new response to process."""
        self.response = response

    def _is_section_heading(self, content: str) -> tuple[bool, bool]:
        """
        Check if content matches section heading patterns.
        Returns (is_section, is_subsection)
        """
        section_pattern = (
            r"^\d+\.\d+\.?\s+\w+"  # e.g., "4.1." or "4.1 Linear Regression"
        )
        subsection_pattern = r"^\d+\.\d+\.\d+\s+\w+"  # e.g., "4.1.1 Basis functions"

        is_subsection = bool(re.match(subsection_pattern, content))
        is_section = bool(re.match(section_pattern, content)) and not is_subsection

        return is_section, is_subsection

    def process_paragraph(self, paragraph: Dict) -> Optional[Dict]:
        """Process a single paragraph and update metadata accordingly."""
        content = paragraph.get("content", "")
        role = paragraph.get("role")
        locations = paragraph.get("locations", [])

        # Get the current page number from the paragraph's location
        current_page = locations[0].get("page", 0) if locations else 0

        # Update metadata based on paragraph type
        if role == "title":
            # Just mark the page as a title page and skip
            self._title_pages.add(current_page)
            return None

        elif role == "sectionHeading" or any(self._is_section_heading(content)):
            is_section, is_subsection = self._is_section_heading(content)
            if is_section and self.section_header != content:
                self.current_metadata.section_heading = content
                self.section_header = content
                self.subsection_header = ""  # Clear subsection when new section starts
                self.current_metadata.subsection_heading = ""
            elif is_subsection:
                self.subsection_header = content  # Update class variable
                self.current_metadata.subsection_heading = content
            return None

        elif role == "pageHeader":
            self.current_metadata.page_header = content
            return None

        elif role == "pageNumber":
            try:
                self.current_metadata.page_number = int(content)
            except ValueError:
                pass
            return None

        # Skip paragraphs on pages with titles
        if current_page in self._title_pages:
            return None

        # Create metadata copy for this paragraph
        paragraph_metadata = self._create_metadata_copy()

        # Use class variable for subsection if not present
        if not paragraph_metadata.subsection_heading and self.subsection_header:
            paragraph_metadata.subsection_heading = self.subsection_header

        # Handle formulas and formula references
        is_formula, formula_ref, clean_content = self._get_formula_reference(content)
        if is_formula:
            i = 1
        if formula_ref:
            if self.formula_buffer:
                # Combine formula buffer content with current content
                buffer_contents = [p["content"] for p in self.formula_buffer]
                buffer_contents.append(clean_content)
                buffer_contents.append(formula_ref)
                combined_content = " ".join(buffer_contents)
                is_formula, formula_ref, clean_content = self._get_formula_reference(
                    combined_content
                )

            if is_formula and not self.formula_buffer:
                # This is a standalone formula
                formula_obj = {
                    "content": clean_content,
                    "formula_reference": formula_ref,
                    "locations": locations,
                }
            elif is_formula and self.formula_buffer:
                formula_obj = {
                    "content": clean_content,
                    "formula_reference": formula_ref,
                    "locations": [
                        loc
                        for p in self.formula_buffer
                        for loc in p.get("locations", [])
                    ],
                }
                locations = [
                    loc for p in self.formula_buffer for loc in p.get("locations", [])
                ]

            if is_formula and formula_obj not in self.formulas:
                formula_metadata = self._create_metadata_copy()
                formula_metadata.formula_references.append(formula_ref)
                self.formulas.append(formula_obj)
                return {
                    "content": clean_content,
                    "metadata": asdict(formula_metadata),
                    "locations": locations,
                }
            else:
                # This is a reference to a formula in text
                paragraph_metadata.formula_references.append(formula_ref)

        # Handle figure references
        figure_ref = self._is_figure_reference(content)
        if figure_ref:
            if content.startswith(figure_ref):
                # This is a figure description
                figure_metadata = self._create_metadata_copy()

                # Check for formula references in figure description
                is_formula, formula_ref, _ = self._get_formula_reference(content)
                if (
                    formula_ref and not is_formula
                ):  # Only add as reference if not a standalone formula
                    figure_metadata.formula_references.append(formula_ref)

                figure_obj = {
                    "figure_reference": figure_ref.replace(" ", "_"),
                    "description": content,
                    "locations": locations,
                    "metadata": asdict(figure_metadata),
                }
                figure_metadata.figure_references.append(figure_ref.replace(" ", "_"))
                if figure_obj not in self.figures:
                    self.figures.append(figure_obj)
                    return {
                        "content": content,
                        "locations": locations,
                        "metadata": asdict(figure_metadata),
                    }
                return {
                    "content": content,
                    "locations": locations,
                    "metadata": asdict(figure_metadata),
                }
            else:
                # This is a reference to a figure in text
                paragraph_metadata.figure_references.append(
                    figure_ref.replace(" ", "_")
                )

        # Handle short paragraphs that might be part of formulas
        if len(content) < 15:
            # Store for potential formula combination
            return None

        # Regular paragraph
        return {
            "content": content,
            "metadata": asdict(paragraph_metadata),
            "locations": locations,
        }

    def _get_formula_reference(
        self, content: str
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Check if content contains formula reference pattern (e.g., '(4.3)').
        Returns (is_formula, formula_ref, clean_content):
            - is_formula: True if this is a standalone formula (at end of text)
            - formula_ref: The formula reference if found, None otherwise
            - clean_content: Content with formula reference removed if it's at the end
        """
        match = re.search(r"\(\d+\.\d+\)(?: \.|\.)?$", content)
        if not match:
            return False, None, content

        formula_ref = match.group(0)
        # It's a formula if it's at the end of the content
        is_formula = match.end() == len(content)

        # If it's a formula, remove the reference from content
        clean_content = content[: match.start()].rstrip() if is_formula else content

        return is_formula, formula_ref, clean_content

    def _is_figure_reference(self, content: str) -> str:
        """Check if content contains figure reference (e.g., 'Figure 4.1')"""
        match = re.search(r"Figure \d+\.\d+", content)
        return match.group(0) if match else None

    def _create_metadata_copy(self) -> DocumentMetadata:
        """Create a copy of current metadata with empty references"""
        return DocumentMetadata(
            page_number=self.current_metadata.page_number,
            section_heading=self.current_metadata.section_heading,
            subsection_heading=self.current_metadata.subsection_heading,
            page_header=self.current_metadata.page_header,
            figure_references=[],
            formula_references=[],
        )

    def save_to_json(self, base_filepath: str) -> None:
        """
        Save processed results to JSON files.

        Args:
            base_filepath: Base path for saving JSON files (without extension)
        """

        # Save documents
        with open(f"{base_filepath}_documents.json", "w", encoding="utf-8") as f:
            json.dump(
                [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in self.documents
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )

        # Save formulas
        with open(f"{base_filepath}_formulas.json", "w", encoding="utf-8") as f:
            json.dump(self.formulas, f, indent=2, ensure_ascii=False)

        # Save figures
        with open(f"{base_filepath}_figures.json", "w", encoding="utf-8") as f:
            json.dump(self.figures, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import os

    # Define paths
    analysis_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "analysis_results"
    )
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "processed_results"
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(analysis_dir, f"paragraphs_data_20241125_132740.json")

    # Load the JSON data
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            response_data = json.load(f)

        print(f"\nProcessing Chapter 4-...")

        # Process the data
        processor = AzureResponseProcessor()
        processor.set_response(response_data)
        documents, formulas, figures = processor.process_paragraphs()

        # Save results
        output_base = os.path.join(output_dir, f"chapter4-")
        processor.save_to_json(output_base)

        print(
            f"Found {len(documents)} documents, {len(formulas)} formulas, and {len(figures)} figures"
        )
        print(f"Results saved to {output_base}_*.json")

    except FileNotFoundError:
        print(f"Warning: Could not find {input_file}")
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in {input_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")
