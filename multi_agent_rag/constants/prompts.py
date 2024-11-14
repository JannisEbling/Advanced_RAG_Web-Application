RESPONSE_PROMPT: str = """You are a RAG solution for the user manual of the TC19, a textile card machine of Trützschler.
                Use the following pieces of context to answer the question or command at the end, only based on the context.
                If you cannot answer based on the context, just say that you don't know, don't try to make up an answer.
                \n\n{context}"""
ALTERNATIVE_RESPONSE_PROMPT: str = """You are a RAG solution for a machine learning book. The last response to the following question contained hallucinations. 
                Say that you have difficulty answering the question on the provided context but answer and explain the question to the best of your knowledge, 
                also tell to factcheck and how the user could do that.
                ---{context}"""
ROUTING_PROMPT: str = """You are an expert at routing a user question to a vectorstore or arxiv.
                    The vectorstore contains book about general theory and practice of machine learning, deeplearning and mathematics.
                    Use the vectorstore for questions on these topics. If the question looks for a specific paper or current research use the arxiv"""
HALLUCINATION_DETECION_PROMPT = """You are a fact-checker that determines if the given answer {answer} is grounded in the given context {context}
                                you don't mind if it doesn't make sense, as long as it is grounded in the context.
                                output a json containing the answer to the question, and appart from the json format don't output any additional text."""
RERANKING_PROMPT = """On a scale of 1-10, rate the relevance of the following document to the query. Consider the specific context and intent of the query, not just keyword matches. Query: {query} Document: {doc} Relevance Score:"""
VECTORSTORE_QUERY_REWRITE_PROMPT = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system that contains a book that covers various machine learning and deep learning topics. 
Given the original query, rewrite it to be more specific, detailed, likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""
ARXIV_QUERY_REWRITE_PROMPT = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system using the arXiv utility from langchain. 
Given the original query, rewrite it to be more specific, detailed, likely to retrieve relevant information.

Original query: {original_query}

Rewritten query:"""
CHUNK_REWRITE_PROMPT = """You are an AI assistant tasked with reformulating chunks of the original text to improve retrieval in a RAG system. 
Given the original chunk, rewrite it to be more likely to be a match for relevant queries.
Ensure the rewritten chunk preserves the key concepts from the original to enhance search relevance and do not add any information that wasnt given in the original chunk. 
Also add a headline to the chunk, use the chunk information and the chapter name for that.

Original chunk: {original_chunk}
Chapter name: {chapter_name}

Result should loo like "<headline>---<rewritten_chunk_content>"
Rewritten chunk with headline:"""
