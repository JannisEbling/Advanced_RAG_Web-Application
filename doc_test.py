import os
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, AnalyzeDocumentRequest

# use your `key` and `endpoint` environment variables
endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")


# helper functions
def get_words(page, line):
    result = []
    for word in page.words:
        if _in_span(word, line.spans):
            result.append(word)
    return result


def _in_span(word, spans):
    for span in spans:
        if word.span.offset >= span.offset and (
            word.span.offset + word.span.length
        ) <= (span.offset + span.length):
            return True
    return False


def analyze_read():
    # sample document
    formUrl = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/rest-api/read.png"

    client = DocumentIntelligenceClient(
        endpoint=endpoint, credential=AzureKeyCredential(key)
    )

    poller = client.begin_analyze_document(
        "prebuilt-read", AnalyzeDocumentRequest(url_source=formUrl)
    )
    result: AnalyzeResult = poller.result()

    print("----Languages detected in the document----")
    if result.languages is not None:
        for language in result.languages:
            print(
                f"Language code: '{language.locale}' with confidence {language.confidence}"
            )

    print("----Styles detected in the document----")
    if result.styles:
        for style in result.styles:
            if style.is_handwritten:
                print("Found the following handwritten content: ")
                print(
                    ",".join(
                        [
                            result.content[span.offset : span.offset + span.length]
                            for span in style.spans
                        ]
                    )
                )
            if style.font_style:
                print(
                    f"The document contains '{style.font_style}' font style, applied to the following text: "
                )
                print(
                    ",".join(
                        [
                            result.content[span.offset : span.offset + span.length]
                            for span in style.spans
                        ]
                    )
                )

    for page in result.pages:
        print(f"----Analyzing document from page #{page.page_number}----")
        print(
            f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}"
        )

        if page.lines:
            for line_idx, line in enumerate(page.lines):
                words = get_words(page, line)
                print(
                    f"...Line # {line_idx} has {len(words)} words and text '{line.content}' within bounding polygon '{line.polygon}'"
                )

                for word in words:
                    print(
                        f"......Word '{word.content}' has a confidence of {word.confidence}"
                    )

        if page.selection_marks:
            for selection_mark in page.selection_marks:
                print(
                    f"...Selection mark is '{selection_mark.state}' within bounding polygon "
                    f"'{selection_mark.polygon}' and has a confidence of {selection_mark.confidence}"
                )

    if result.paragraphs:
        print(f"----Detected #{len(result.paragraphs)} paragraphs in the document----")
        for paragraph in result.paragraphs:
            print(
                f"Found paragraph with role: '{paragraph.role}' within {paragraph.bounding_regions} bounding region"
            )
            print(f"...with content: '{paragraph.content}'")

        result.paragraphs.sort(
            key=lambda p: (p.spans.sort(key=lambda s: s.offset), p.spans[0].offset)
        )
        print("-----Print sorted paragraphs-----")
        for idx, paragraph in enumerate(result.paragraphs):
            print(
                f"...paragraph:{idx} with offset: {paragraph.spans[0].offset} and length: {paragraph.spans[0].length}"
            )

    print("----------------------------------------")




if __name__ == "__main__":
    analyze_read()

