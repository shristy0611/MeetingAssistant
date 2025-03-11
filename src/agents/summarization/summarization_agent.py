'''Summarization Agent for AMPTALK.

This agent performs meeting summarization using state-of-the-art transformer models. It leverages
advanced NLP techniques for meeting summarization, action item extraction, key point identification,
timeline generation, and priority tagging.
'''

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    pipeline,
    T5ForConditionalGeneration,
    BartForConditionalGeneration
)

from src.core.agent import Agent
from src.core.message import Message

logger = logging.getLogger(__name__)


@dataclass
class SummarizationConfig:
    """Configuration for the Summarization Agent."""
    
    # General summarization
    summarization_model_name: str = "knkarthick/MEETING_SUMMARY"
    max_summary_length: int = 150
    min_summary_length: int = 30
    
    # Action item extraction
    action_item_model_name: str = "facebook/bart-large-mnli"
    action_item_threshold: float = 0.7
    action_item_keywords: List[str] = field(default_factory=lambda: [
        "need to", "should", "will", "going to", "have to", "must", 
        "action", "task", "todo", "to-do", "follow up", "followup", 
        "follow-up", "deadline", "by tomorrow", "by next week", 
        "by monday", "by tuesday", "by wednesday", "by thursday", 
        "by friday", "by saturday", "by sunday"
    ])
    
    # Key point identification
    key_point_model_name: str = "facebook/bart-large-mnli"
    key_point_threshold: float = 0.6
    max_key_points: int = 5
    
    # Timeline generation
    timeline_window_size: int = 5  # Number of segments to consider for timeline
    timeline_min_segment_length: int = 100  # Minimum characters for a timeline segment
    
    # Priority tagging
    priority_levels: List[str] = field(default_factory=lambda: ["High", "Medium", "Low"])
    priority_model_name: str = "facebook/bart-large-mnli"
    priority_threshold: float = 0.6
    
    # Topic segmentation
    topic_segmentation_method: str = "cosine"  # "linear", "cosine", or "complex_cosine"
    min_segment_length: int = 200  # Minimum characters for a segment
    max_segment_length: int = 1000  # Maximum characters for a segment
    
    # General settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    cache_dir: str = "model_cache"


class SummarizationAgent(Agent):
    """
    Summarization Agent that provides comprehensive meeting summarization capabilities.
    
    Capabilities include:
    - Meeting summarization
    - Action item extraction
    - Key point identification
    - Timeline generation
    - Priority tagging
    """
    
    def __init__(
        self, 
        agent_id: str, 
        config: Optional[SummarizationConfig] = None
    ):
        super().__init__(agent_id=agent_id)
        self.agent_id = agent_id
        self.config = config or SummarizationConfig()
        
        logger.info(f"Initializing Summarization Agent with models: "
                   f"summarization={self.config.summarization_model_name}")
        
        # Initialize summarization model
        self.summarizer = pipeline(
            "summarization", 
            model=self.config.summarization_model_name,
            device=0 if self.config.device == "cuda" else -1
        )
        
        # Initialize zero-shot classification for action items and key points
        self.zero_shot_classifier = pipeline(
            "zero-shot-classification",
            model=self.config.action_item_model_name,
            device=0 if self.config.device == "cuda" else -1
        )
        
        # Initialize tokenizer for topic segmentation
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize sentence transformer for topic segmentation
        self.sentence_transformer = pipeline(
            "feature-extraction",
            model="sentence-transformers/all-MiniLM-L6-v2",
            device=0 if self.config.device == "cuda" else -1
        )
    
    def summarize_meeting(self, text: str) -> dict:
        """Generate a comprehensive meeting summary."""
        # First, segment the text into topics
        segments = self._segment_text_by_topic(text)
        
        # Summarize each segment
        segment_summaries = []
        for segment in segments:
            summary = self.summarizer(
                segment,
                max_length=self.config.max_summary_length,
                min_length=self.config.min_summary_length,
                do_sample=False
            )[0]["summary_text"]
            segment_summaries.append(summary)
        
        # Generate overall summary
        if len(segment_summaries) > 1:
            combined_segments = " ".join(segment_summaries)
            overall_summary = self.summarizer(
                combined_segments,
                max_length=self.config.max_summary_length,
                min_length=self.config.min_summary_length,
                do_sample=False
            )[0]["summary_text"]
        else:
            overall_summary = segment_summaries[0] if segment_summaries else ""
        
        # Extract action items
        action_items = self.extract_action_items(text)
        
        # Identify key points
        key_points = self.identify_key_points(text)
        
        # Generate timeline
        timeline = self.generate_timeline(text)
        
        # Tag priorities
        priorities = self.tag_priorities(action_items)
        
        return {
            "summary": overall_summary,
            "segment_summaries": segment_summaries,
            "action_items": action_items,
            "key_points": key_points,
            "timeline": timeline,
            "priorities": priorities
        }
    
    def extract_action_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract action items from meeting text."""
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        # Filter sentences that might contain action items based on keywords
        potential_action_items = []
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in self.config.action_item_keywords):
                potential_action_items.append(sentence)
        
        if not potential_action_items:
            return []
        
        # Use zero-shot classification to identify action items
        action_items = []
        for sentence in potential_action_items:
            result = self.zero_shot_classifier(
                sentence,
                candidate_labels=["action item", "not action item"],
                multi_label=False
            )
            
            if result["labels"][0] == "action item" and result["scores"][0] >= self.config.action_item_threshold:
                # Extract assignee and deadline if available
                assignee = self._extract_assignee(sentence)
                deadline = self._extract_deadline(sentence)
                
                action_items.append({
                    "text": sentence,
                    "confidence": result["scores"][0],
                    "assignee": assignee,
                    "deadline": deadline
                })
        
        return action_items
    
    def identify_key_points(self, text: str) -> List[Dict[str, Any]]:
        """Identify key points from meeting text."""
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        
        # Use zero-shot classification to identify key points
        key_points = []
        for paragraph in paragraphs:
            if len(paragraph.strip()) < 20:  # Skip very short paragraphs
                continue
                
            result = self.zero_shot_classifier(
                paragraph,
                candidate_labels=["key point", "not key point"],
                multi_label=False
            )
            
            if result["labels"][0] == "key point" and result["scores"][0] >= self.config.key_point_threshold:
                # Summarize the key point to make it concise
                summary = self.summarizer(
                    paragraph,
                    max_length=50,
                    min_length=10,
                    do_sample=False
                )[0]["summary_text"]
                
                key_points.append({
                    "text": summary,
                    "original_text": paragraph,
                    "confidence": result["scores"][0]
                })
        
        # Sort by confidence and take top N
        key_points = sorted(key_points, key=lambda x: x["confidence"], reverse=True)
        return key_points[:self.config.max_key_points]
    
    def generate_timeline(self, text: str) -> List[Dict[str, Any]]:
        """Generate a timeline of the meeting."""
        # Segment the text
        segments = self._segment_text_by_topic(text)
        
        # Generate timeline entries
        timeline = []
        for i, segment in enumerate(segments):
            # Skip segments that are too short
            if len(segment) < self.config.timeline_min_segment_length:
                continue
                
            # Summarize the segment for the timeline
            summary = self.summarizer(
                segment,
                max_length=30,
                min_length=10,
                do_sample=False
            )[0]["summary_text"]
            
            # Extract any time references
            time_references = self._extract_time_references(segment)
            
            timeline.append({
                "position": i + 1,
                "summary": summary,
                "time_references": time_references
            })
        
        return timeline
    
    def tag_priorities(self, action_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Tag priorities for action items."""
        if not action_items:
            return []
            
        prioritized_items = []
        for item in action_items:
            result = self.zero_shot_classifier(
                item["text"],
                candidate_labels=self.config.priority_levels,
                multi_label=False
            )
            
            priority = result["labels"][0]
            confidence = result["scores"][0]
            
            prioritized_items.append({
                **item,
                "priority": priority,
                "priority_confidence": confidence
            })
        
        return prioritized_items
    
    def _segment_text_by_topic(self, text: str) -> List[str]:
        """Segment text by topic using the specified method."""
        if self.config.topic_segmentation_method == "linear":
            return self._linear_segmentation(text)
        elif self.config.topic_segmentation_method == "cosine":
            return self._cosine_segmentation(text)
        elif self.config.topic_segmentation_method == "complex_cosine":
            return self._complex_cosine_segmentation(text)
        else:
            logger.warning(f"Unknown segmentation method: {self.config.topic_segmentation_method}. Using linear segmentation.")
            return self._linear_segmentation(text)
    
    def _linear_segmentation(self, text: str) -> List[str]:
        """Segment text linearly based on character count."""
        if len(text) <= self.config.max_segment_length:
            return [text]
            
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        segments = []
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) <= self.config.max_segment_length:
                current_segment += sentence + " "
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + " "
        
        if current_segment:
            segments.append(current_segment.strip())
            
        return segments
    
    def _cosine_segmentation(self, text: str) -> List[str]:
        """Segment text using cosine similarity between sentence embeddings."""
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 1:
            return [text]
            
        # Get embeddings for each sentence
        embeddings = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            embedding = self.sentence_transformer(sentence)
            embeddings.append(embedding[0][0])  # Get the CLS token embedding
            
        if len(embeddings) <= 1:
            return [text]
            
        # Calculate cosine similarities between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)
            
        # Find segment boundaries (local minima in similarity)
        boundaries = []
        for i in range(1, len(similarities) - 1):
            if similarities[i] < similarities[i - 1] and similarities[i] < similarities[i + 1]:
                boundaries.append(i + 1)  # +1 because we want the index of the sentence after the boundary
                
        # Ensure segments are not too small or too large
        filtered_boundaries = []
        last_boundary = 0
        for boundary in boundaries:
            segment_size = boundary - last_boundary
            if segment_size >= 3:  # Minimum 3 sentences per segment
                filtered_boundaries.append(boundary)
                last_boundary = boundary
                
        # Create segments based on boundaries
        segments = []
        start_idx = 0
        for boundary in filtered_boundaries:
            segment = " ".join(sentences[start_idx:boundary])
            if len(segment) >= self.config.min_segment_length:
                segments.append(segment)
            start_idx = boundary
            
        # Add the last segment
        last_segment = " ".join(sentences[start_idx:])
        if len(last_segment) >= self.config.min_segment_length:
            segments.append(last_segment)
            
        # If no segments were created, return the original text
        if not segments:
            return [text]
            
        return segments
    
    def _complex_cosine_segmentation(self, text: str) -> List[str]:
        """Advanced segmentation using sliding window and dynamic programming."""
        # This is a simplified version of the algorithm described in the paper
        # "Action-Item-Driven Summarization of Long Meeting Transcripts"
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        if len(sentences) <= 5:  # Too short to segment
            return [text]
            
        # Get embeddings for each sentence
        embeddings = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            embedding = self.sentence_transformer(sentence)
            embeddings.append(embedding[0][0])  # Get the CLS token embedding
            
        if len(embeddings) <= 5:
            return [text]
            
        # Calculate similarity matrix
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                similarity_matrix[i, j] = self._cosine_similarity(embeddings[i], embeddings[j])
                
        # Use dynamic programming to find optimal segmentation
        # This is a simplified version that looks for natural breaks in the text
        window_size = 5  # Look at 5 sentences at a time
        boundaries = []
        
        for i in range(window_size, len(embeddings) - window_size):
            # Calculate average similarity within windows before and after current position
            before_avg = np.mean([similarity_matrix[j, k] for j in range(i - window_size, i) for k in range(i - window_size, i) if j != k])
            after_avg = np.mean([similarity_matrix[j, k] for j in range(i, i + window_size) for k in range(i, i + window_size) if j != k])
            
            # Calculate average similarity between windows
            between_avg = np.mean([similarity_matrix[j, k] for j in range(i - window_size, i) for k in range(i, i + window_size)])
            
            # If similarity within windows is high but between windows is low, it's a boundary
            if before_avg > 0.5 and after_avg > 0.5 and between_avg < 0.3:
                boundaries.append(i)
                
        # Ensure boundaries are not too close
        filtered_boundaries = []
        last_boundary = 0
        for boundary in boundaries:
            if boundary - last_boundary >= 5:  # At least 5 sentences between boundaries
                filtered_boundaries.append(boundary)
                last_boundary = boundary
                
        # Create segments based on boundaries
        segments = []
        start_idx = 0
        for boundary in filtered_boundaries:
            segment = " ".join(sentences[start_idx:boundary])
            if len(segment) >= self.config.min_segment_length:
                segments.append(segment)
            start_idx = boundary
            
        # Add the last segment
        last_segment = " ".join(sentences[start_idx:])
        if len(last_segment) >= self.config.min_segment_length:
            segments.append(last_segment)
            
        # If no segments were created, return the original text
        if not segments:
            return [text]
            
        return segments
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with a more sophisticated approach
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() + " " for s in sentences if s.strip()]
    
    def _extract_assignee(self, text: str) -> Optional[str]:
        """Extract the assignee from an action item."""
        # Look for common patterns like "John will...", "Mary needs to..."
        patterns = [
            r'([A-Z][a-z]+) (will|should|needs to|has to|must|is going to)',
            r'([A-Z][a-z]+) (is|are) responsible for',
            r'assigned to ([A-Z][a-z]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
                
        return None
    
    def _extract_deadline(self, text: str) -> Optional[str]:
        """Extract the deadline from an action item."""
        # Look for common deadline patterns
        patterns = [
            r'by (tomorrow|next week|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'due (tomorrow|next week|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'by (January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}',
            r'by \d{1,2}/\d{1,2}(/\d{2,4})?'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
                
        return None
    
    def _extract_time_references(self, text: str) -> List[str]:
        """Extract time references from text."""
        # Look for time references like "at 2pm", "10 minutes ago", etc.
        patterns = [
            r'at \d{1,2}(:\d{2})? ?(am|pm)',
            r'\d{1,2}(:\d{2})? ?(am|pm)',
            r'\d+ (minutes|hours) (ago|later)',
            r'(earlier|later) (today|this morning|this afternoon|this evening)'
        ]
        
        time_references = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        time_references.append(''.join(match).strip())
                    else:
                        time_references.append(match.strip())
                
        return time_references
    
    async def handle_summarize(self, message: Message) -> Message:
        """Handle a summarization request."""
        text = message.data.get("text", "")
        if not text:
            return Message(
                sender=self.agent_id, 
                receiver=message.sender, 
                type="error", 
                data={"error": "No text provided for summarization"}
            )
        
        result = self.summarize_meeting(text)
        return Message(
            sender=self.agent_id, 
            receiver=message.sender, 
            type="summarization_results", 
            data={"result": result, "original_text": text}
        )
    
    async def handle_extract_action_items(self, message: Message) -> Message:
        """Handle an action item extraction request."""
        text = message.data.get("text", "")
        if not text:
            return Message(
                sender=self.agent_id, 
                receiver=message.sender, 
                type="error", 
                data={"error": "No text provided for action item extraction"}
            )
        
        result = self.extract_action_items(text)
        return Message(
            sender=self.agent_id, 
            receiver=message.sender, 
            type="action_item_results", 
            data={"result": result, "original_text": text}
        )
    
    async def handle_identify_key_points(self, message: Message) -> Message:
        """Handle a key point identification request."""
        text = message.data.get("text", "")
        if not text:
            return Message(
                sender=self.agent_id, 
                receiver=message.sender, 
                type="error", 
                data={"error": "No text provided for key point identification"}
            )
        
        result = self.identify_key_points(text)
        return Message(
            sender=self.agent_id, 
            receiver=message.sender, 
            type="key_point_results", 
            data={"result": result, "original_text": text}
        )
    
    async def handle_generate_timeline(self, message: Message) -> Message:
        """Handle a timeline generation request."""
        text = message.data.get("text", "")
        if not text:
            return Message(
                sender=self.agent_id, 
                receiver=message.sender, 
                type="error", 
                data={"error": "No text provided for timeline generation"}
            )
        
        result = self.generate_timeline(text)
        return Message(
            sender=self.agent_id, 
            receiver=message.sender, 
            type="timeline_results", 
            data={"result": result, "original_text": text}
        )
    
    async def handle_tag_priorities(self, message: Message) -> Message:
        """Handle a priority tagging request."""
        action_items = message.data.get("action_items", [])
        if not action_items:
            return Message(
                sender=self.agent_id, 
                receiver=message.sender, 
                type="error", 
                data={"error": "No action items provided for priority tagging"}
            )
        
        result = self.tag_priorities(action_items)
        return Message(
            sender=self.agent_id, 
            receiver=message.sender, 
            type="priority_results", 
            data={"result": result}
        )
    
    async def shutdown(self):
        """Shutdown the agent."""
        logger.info(f"Shutting down Summarization Agent: {self.agent_id}")
        await super().shutdown() 