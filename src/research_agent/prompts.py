router_instructions = """
You are an expert at routing a user question to a vectorstore or web search.

The vectorstore contains documents related to huggingface pipeline objects only, it doesn't contain any other huggingface documents or other things.

Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

query_writer_instructions = """Your goal is to generate a targeted web search query.
The query will gather information related to a specific topic.

<TOPIC>
{research_topic}
</TOPIC>

<FORMAT>
Format your response as a JSON object with ALL three of these exact keys:
   - "query": The actual search query string
   - "aspect": The specific aspect of the topic being researched
   - "rationale": Brief explanation of why this query is relevant
</FORMAT>

<EXAMPLE>
Example output:
{{
    "query": "machine learning transformer architecture explained",
    "aspect": "technical architecture",
    "rationale": "Understanding the fundamental structure of transformer models"
}}
</EXAMPLE>

Provide your response in JSON format:"""

web_summarizer_instructions = """
<GOAL>
Generate a high-quality summary of the web search results and keep it concise / related to the user topic.
</GOAL>

<REQUIREMENTS>
When creating a NEW summary:
1. Highlight the most relevant information related to the user topic from the search results
2. Ensure a coherent flow of information

When EXTENDING an existing summary:                                                                                                                 
1. Read the existing summary and new search results carefully.                                                    
2. Compare the new information with the existing summary.                                                         
3. For each piece of new information:                                                                             
    a. If it's related to existing points, integrate it into the relevant paragraph.                               
    b. If it's entirely new but relevant, add a new paragraph with a smooth transition.                            
    c. If it's not relevant to the user topic, skip it.                                                            
4. Ensure all additions are relevant to the user's topic.                                                         
5. Verify that your final output differs from the input summary.                                                                                                                                                            
< /REQUIREMENTS >

< FORMATTING >
- Start directly with the updated summary, without preamble or titles. Do not use XML tags in the output.  
< /FORMATTING >"""

reflection_instructions = """You are an expert research assistant analyzing a summary about {research_topic}. Your task is to assess the quality and completeness of the summary while identifying any knowledge gaps, particularly discrepancies or missing details between web sources and RAG (Retrieval-Augmented Generation) results.

<GOAL>
1. Identify knowledge gaps or areas that need deeper exploration.
2. If both web summaries and RAG results exist:
   - Compare the information provided by web sources and RAG results.
   - Highlight discrepancies or contradictions, if any.
   - Identify missing insights that could improve factual accuracy or completeness.
3. If only web summaries exist:
   - Assess whether the web-based information is thorough.
   - Identify missing technical details or new advancements that require further exploration.
4. Generate a follow-up question that would help expand the understanding of the topic.
5. Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

<REQUIREMENTS>
- Ensure the follow-up question is self-contained and includes necessary context for web search.
- Avoid unnecessary speculation; base the knowledge gap analysis on the provided summary.

<FORMAT>
Format your response as a JSON object with these exact keys:
- **"knowledge_gap"**: Describe what information is missing or needs clarification.
- **"source_discrepancy"**: If both web and RAG results exist, highlight any conflicts or missing details between them. If not applicable, return `"None"`.
- **"follow_up_query"**: Write a specific question to address this gap.

<EXAMPLES>
Example 1: (Both web and RAG results exist, with discrepancies)
---
{{
    "knowledge_gap": "The summary does not clarify the differences between LoRA and full fine-tuning efficiency.",
    "source_discrepancy": "The web sources emphasize LoRA as the best approach, while RAG documentation suggests full fine-tuning may still be beneficial for some tasks.",
    "follow_up_query": "What are the trade-offs between LoRA and full fine-tuning for transformer models?"
}}

Example 2: (Only web summaries exist)
---
{{
    "knowledge_gap": "The summary lacks insights on hardware acceleration techniques for model inference.",
    "source_discrepancy": "None",
    "follow_up_query": "What are the best hardware acceleration techniques for optimizing transformer inference?"
}}

</EXAMPLES>

Provide your analysis in JSON format."""
