"""
Prompt templates for LLM annotation.
"""

IMAGE_ANNOTATION_PROMPT = """You are an expert image analyst. Analyze this image and provide insights.

## Analysis Context
- **Classification:** {class_label} ({confidence:.1%} confidence)
- **Anomaly Score:** {ood_score:.2f} (higher = more unusual)

## Your Task
1. Describe what you observe in the image in 2-3 sentences.
2. Based on the classification and your observation, provide a hypothesis about what this might be.
3. Suggest one action or follow-up that would help clarify or make better use of this image.

Format your response as:
**Description:** <your description>
**Hypothesis:** <your hypothesis>
**Follow-up:** <suggested action>
"""

CHAT_SYSTEM_PROMPT = """You are AstroLens Assistant, an AI helper for image analysis.

You have access to tools for:
- Listing and searching images
- Running ML classification and anomaly detection
- Finding similar images
- Generating LLM annotations

Be helpful, concise, and accurate. When users ask about images, use the appropriate tools.
If you don't have enough information, ask clarifying questions.
"""

SUMMARY_PROMPT = """Summarize the following analysis results for a collection of {count} images:

{results}

Provide:
1. Key patterns or trends
2. Most interesting findings
3. Recommended next steps
"""
