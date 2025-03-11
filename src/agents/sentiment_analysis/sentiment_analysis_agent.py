'''Sentiment Analysis Agent for AMPTALK.

This agent performs sentiment analysis using state-of-the-art models. It leverages Transformer-based models for detecting sentiment, emotion tracking, pain point identification, context-aware analysis, and trend analysis.
'''

import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForTokenClassification,
    pipeline
)

from src.core.agent import Agent
from src.core.message import Message

logger = logging.getLogger(__name__)


@dataclass
class SentimentAnalysisConfig:
    """Configuration for the Sentiment Analysis Agent."""
    
    # Sentiment analysis
    sentiment_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Emotion tracking
    emotion_model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    emotion_threshold: float = 0.3
    
    # Pain point identification
    pain_point_model_name: str = "distilbert-base-uncased"
    pain_point_keywords: List[str] = field(default_factory=lambda: [
        "issue", "problem", "disappointed", "frustrating", "difficult", 
        "broken", "doesn't work", "not working", "failed", "failure",
        "bug", "error", "crash", "slow", "expensive", "overpriced",
        "waste", "poor", "terrible", "horrible", "awful", "bad"
    ])
    
    # Context-aware analysis
    context_window_size: int = 5  # Number of previous messages to consider
    
    # Trend analysis
    trend_window_size: int = 20  # Number of data points to consider for trend analysis
    trend_smoothing_factor: float = 0.3  # Exponential smoothing factor (0-1)
    trend_change_threshold: float = 0.15  # Threshold for significant trend change
    
    # General settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    cache_dir: str = "model_cache"


class SentimentAnalysisAgent(Agent):
    """
    Sentiment Analysis Agent that provides comprehensive sentiment analysis capabilities.
    
    Capabilities include:
    - Real-time sentiment detection
    - Emotion tracking
    - Pain point identification
    - Context-aware analysis
    - Trend analysis
    """
    
    def __init__(
        self, 
        agent_id: str, 
        config: Optional[SentimentAnalysisConfig] = None
    ):
        super().__init__(agent_id=agent_id)
        self.agent_id = agent_id
        self.config = config or SentimentAnalysisConfig()
        
        logger.info(f"Initializing Sentiment Analysis Agent with models: "
                   f"sentiment={self.config.sentiment_model_name}, "
                   f"emotion={self.config.emotion_model_name}")
        
        # Initialize sentiment model
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.config.sentiment_model_name)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.config.sentiment_model_name)
        self.sentiment_model.eval()
        
        # Initialize emotion model
        self.emotion_tokenizer = AutoTokenizer.from_pretrained(self.config.emotion_model_name)
        self.emotion_model = AutoModelForSequenceClassification.from_pretrained(self.config.emotion_model_name)
        self.emotion_model.eval()
        
        # Initialize pain point model (using zero-shot classification)
        self.pain_point_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=0 if self.config.device == "cuda" else -1
        )
        
        # Context storage
        self.conversation_history = {}
        
        # Trend analysis storage
        self.sentiment_history = {}
        self.emotion_history = {}
        self.pain_point_history = {}
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze the sentiment of a text."""
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        sentiment = "positive" if scores[1] > scores[0] else "negative"
        confidence = scores[1].item() if sentiment == "positive" else scores[0].item()
        return {"sentiment": sentiment, "confidence": confidence, "scores": scores.tolist()}
    
    def track_emotions(self, text: str) -> dict:
        """Track emotions in a text using a multi-label emotion classifier."""
        inputs = self.emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.emotion_model(**inputs)
        
        scores = torch.sigmoid(outputs.logits)[0]
        emotions = self.emotion_model.config.id2label
        
        # Get emotions above threshold
        detected_emotions = []
        for idx, score in enumerate(scores):
            if score.item() > self.config.emotion_threshold:
                detected_emotions.append({
                    "emotion": emotions[idx],
                    "score": score.item()
                })
        
        # Sort by score
        detected_emotions = sorted(detected_emotions, key=lambda x: x["score"], reverse=True)
        
        return {
            "emotions": detected_emotions,
            "dominant_emotion": detected_emotions[0]["emotion"] if detected_emotions else None,
            "all_scores": {emotions[i]: score.item() for i, score in enumerate(scores)}
        }
    
    def identify_pain_points(self, text: str) -> dict:
        """Identify pain points in customer feedback."""
        # Use zero-shot classification to identify pain points
        pain_point_categories = [
            "product issue", 
            "service complaint", 
            "pricing concern", 
            "usability problem",
            "performance issue",
            "feature request"
        ]
        
        results = self.pain_point_pipeline(text, pain_point_categories, multi_label=True)
        
        # Extract pain points above threshold (0.5)
        detected_pain_points = []
        for label, score in zip(results["labels"], results["scores"]):
            if score > 0.5:
                detected_pain_points.append({"category": label, "confidence": score})
        
        # Also check for pain point keywords
        keyword_matches = []
        for keyword in self.config.pain_point_keywords:
            if keyword.lower() in text.lower():
                keyword_matches.append(keyword)
        
        return {
            "pain_points": detected_pain_points,
            "keyword_matches": keyword_matches,
            "has_pain_points": len(detected_pain_points) > 0 or len(keyword_matches) > 0
        }
    
    def analyze_sentiment_trend(self, conversation_id: str) -> dict:
        """Analyze sentiment trends over time for a specific conversation."""
        if conversation_id not in self.sentiment_history or len(self.sentiment_history[conversation_id]) < 2:
            return {
                "trend": "insufficient_data",
                "change_points": [],
                "sentiment_scores": [],
                "smoothed_scores": []
            }
        
        # Get sentiment history
        sentiment_scores = self.sentiment_history[conversation_id]
        
        # Apply exponential smoothing to reduce noise
        smoothed_scores = self._apply_exponential_smoothing(sentiment_scores)
        
        # Detect trend direction
        trend_direction = self._detect_trend_direction(smoothed_scores)
        
        # Detect significant change points
        change_points = self._detect_change_points(smoothed_scores)
        
        # Calculate trend statistics
        trend_stats = self._calculate_trend_statistics(smoothed_scores)
        
        return {
            "trend": trend_direction,
            "change_points": change_points,
            "sentiment_scores": sentiment_scores,
            "smoothed_scores": smoothed_scores,
            "statistics": trend_stats
        }
    
    def analyze_emotion_trend(self, conversation_id: str, emotion_type: str = None) -> dict:
        """Analyze emotion trends over time for a specific conversation."""
        if conversation_id not in self.emotion_history or len(self.emotion_history[conversation_id]) < 2:
            return {
                "trend": "insufficient_data",
                "emotions": {}
            }
        
        # Get emotion history
        emotion_history = self.emotion_history[conversation_id]
        
        # Analyze trends for each emotion type
        emotion_trends = {}
        all_emotions = set()
        
        # Collect all emotion types
        for entry in emotion_history:
            for emotion in entry["emotions"]:
                all_emotions.add(emotion["emotion"])
        
        # Analyze trend for each emotion
        for emotion in all_emotions:
            if emotion_type and emotion != emotion_type:
                continue
                
            # Extract scores for this emotion
            scores = []
            for entry in emotion_history:
                score = 0.0
                for e in entry["emotions"]:
                    if e["emotion"] == emotion:
                        score = e["score"]
                        break
                scores.append(score)
            
            # Apply exponential smoothing
            smoothed_scores = self._apply_exponential_smoothing(scores)
            
            # Detect trend direction
            trend_direction = self._detect_trend_direction(smoothed_scores)
            
            # Detect significant change points
            change_points = self._detect_change_points(smoothed_scores)
            
            emotion_trends[emotion] = {
                "trend": trend_direction,
                "change_points": change_points,
                "scores": scores,
                "smoothed_scores": smoothed_scores
            }
        
        return {
            "trend": "analyzed",
            "emotions": emotion_trends
        }
    
    def analyze_pain_point_trend(self, conversation_id: str) -> dict:
        """Analyze pain point trends over time for a specific conversation."""
        if conversation_id not in self.pain_point_history or len(self.pain_point_history[conversation_id]) < 2:
            return {
                "trend": "insufficient_data",
                "categories": {}
            }
        
        # Get pain point history
        pain_point_history = self.pain_point_history[conversation_id]
        
        # Analyze trends for each pain point category
        category_trends = {}
        all_categories = set()
        
        # Collect all pain point categories
        for entry in pain_point_history:
            for pain_point in entry["pain_points"]:
                all_categories.add(pain_point["category"])
        
        # Analyze trend for each category
        for category in all_categories:
            # Extract scores for this category
            scores = []
            for entry in pain_point_history:
                score = 0.0
                for pp in entry["pain_points"]:
                    if pp["category"] == category:
                        score = pp["confidence"]
                        break
                scores.append(score)
            
            # Apply exponential smoothing
            smoothed_scores = self._apply_exponential_smoothing(scores)
            
            # Detect trend direction
            trend_direction = self._detect_trend_direction(smoothed_scores)
            
            # Detect significant change points
            change_points = self._detect_change_points(smoothed_scores)
            
            category_trends[category] = {
                "trend": trend_direction,
                "change_points": change_points,
                "scores": scores,
                "smoothed_scores": smoothed_scores
            }
        
        # Count keyword occurrences over time
        keyword_trends = {}
        for keyword in self.config.pain_point_keywords:
            counts = []
            for entry in pain_point_history:
                count = entry["keyword_matches"].count(keyword)
                counts.append(count)
            
            if sum(counts) > 0:  # Only include keywords that appeared at least once
                keyword_trends[keyword] = {
                    "counts": counts,
                    "total": sum(counts)
                }
        
        return {
            "trend": "analyzed",
            "categories": category_trends,
            "keywords": keyword_trends
        }
    
    def _apply_exponential_smoothing(self, scores: List[float]) -> List[float]:
        """Apply exponential smoothing to a list of scores."""
        alpha = self.config.trend_smoothing_factor
        smoothed = [scores[0]]  # First value is same as original
        
        for i in range(1, len(scores)):
            smoothed_val = alpha * scores[i] + (1 - alpha) * smoothed[i-1]
            smoothed.append(smoothed_val)
        
        return smoothed
    
    def _detect_trend_direction(self, smoothed_scores: List[float]) -> str:
        """Detect the direction of a trend based on smoothed scores."""
        if len(smoothed_scores) < 2:
            return "insufficient_data"
        
        # Use linear regression to determine trend
        x = np.array(range(len(smoothed_scores)))
        y = np.array(smoothed_scores)
        
        # Calculate slope using least squares
        n = len(x)
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
        
        # Determine trend direction based on slope
        if abs(slope) < 0.01:  # Very small slope
            return "stable"
        elif slope > 0:
            return "improving" if smoothed_scores[-1] > 0 else "deteriorating"
        else:
            return "deteriorating" if smoothed_scores[-1] > 0 else "improving"
    
    def _detect_change_points(self, smoothed_scores: List[float]) -> List[int]:
        """Detect significant change points in the trend."""
        if len(smoothed_scores) < 3:
            return []
        
        change_points = []
        threshold = self.config.trend_change_threshold
        
        for i in range(1, len(smoothed_scores) - 1):
            # Calculate rate of change
            prev_change = smoothed_scores[i] - smoothed_scores[i-1]
            next_change = smoothed_scores[i+1] - smoothed_scores[i]
            
            # Check for sign change with significant magnitude
            if (prev_change * next_change < 0) and (abs(prev_change) > threshold or abs(next_change) > threshold):
                change_points.append(i)
        
        return change_points
    
    def _calculate_trend_statistics(self, scores: List[float]) -> dict:
        """Calculate statistical measures for the trend."""
        if not scores:
            return {}
        
        return {
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std_dev": np.std(scores),
            "min": min(scores),
            "max": max(scores),
            "current": scores[-1],
            "change_from_start": scores[-1] - scores[0] if len(scores) > 1 else 0
        }
    
    def _update_sentiment_history(self, conversation_id: str, sentiment_result: dict):
        """Update the sentiment history for trend analysis."""
        if conversation_id not in self.sentiment_history:
            self.sentiment_history[conversation_id] = []
        
        # Convert sentiment to numeric score (-1 to 1)
        sentiment_score = sentiment_result["confidence"]
        if sentiment_result["sentiment"] == "negative":
            sentiment_score = -sentiment_score
        
        # Add to history
        self.sentiment_history[conversation_id].append(sentiment_score)
        
        # Limit history size
        if len(self.sentiment_history[conversation_id]) > self.config.trend_window_size:
            self.sentiment_history[conversation_id] = self.sentiment_history[conversation_id][-self.config.trend_window_size:]
    
    def _update_emotion_history(self, conversation_id: str, emotion_result: dict):
        """Update the emotion history for trend analysis."""
        if conversation_id not in self.emotion_history:
            self.emotion_history[conversation_id] = []
        
        # Add to history
        self.emotion_history[conversation_id].append(emotion_result)
        
        # Limit history size
        if len(self.emotion_history[conversation_id]) > self.config.trend_window_size:
            self.emotion_history[conversation_id] = self.emotion_history[conversation_id][-self.config.trend_window_size:]
    
    def _update_pain_point_history(self, conversation_id: str, pain_point_result: dict):
        """Update the pain point history for trend analysis."""
        if conversation_id not in self.pain_point_history:
            self.pain_point_history[conversation_id] = []
        
        # Add to history
        self.pain_point_history[conversation_id].append(pain_point_result)
        
        # Limit history size
        if len(self.pain_point_history[conversation_id]) > self.config.trend_window_size:
            self.pain_point_history[conversation_id] = self.pain_point_history[conversation_id][-self.config.trend_window_size:]
    
    def analyze_with_context(self, text: str, conversation_id: str) -> dict:
        """Perform context-aware sentiment analysis."""
        # Update conversation history
        self._update_context(conversation_id, text)
        
        # Get context
        context = self._get_context(conversation_id)
        
        # Analyze current message
        sentiment_result = self.analyze_sentiment(text)
        emotion_result = self.track_emotions(text)
        pain_point_result = self.identify_pain_points(text)
        
        # Update trend history
        self._update_sentiment_history(conversation_id, sentiment_result)
        self._update_emotion_history(conversation_id, emotion_result)
        self._update_pain_point_history(conversation_id, pain_point_result)
        
        # Analyze sentiment trend
        sentiment_trend = "stable"
        if len(context) > 1:
            # Analyze previous message
            prev_sentiment = self.analyze_sentiment(context[-2])
            
            # Determine trend
            if sentiment_result["sentiment"] == "positive" and prev_sentiment["sentiment"] == "negative":
                sentiment_trend = "improving"
            elif sentiment_result["sentiment"] == "negative" and prev_sentiment["sentiment"] == "positive":
                sentiment_trend = "deteriorating"
        
        return {
            "sentiment": sentiment_result,
            "emotions": emotion_result,
            "pain_points": pain_point_result,
            "context_size": len(context),
            "sentiment_trend": sentiment_trend
        }
    
    def _update_context(self, conversation_id: str, text: str):
        """Update the conversation context."""
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []
        
        self.conversation_history[conversation_id].append(text)
        
        # Limit context window size
        if len(self.conversation_history[conversation_id]) > self.config.context_window_size:
            self.conversation_history[conversation_id] = self.conversation_history[conversation_id][-self.config.context_window_size:]
    
    def _get_context(self, conversation_id: str) -> List[str]:
        """Get the conversation context."""
        return self.conversation_history.get(conversation_id, [])
    
    async def handle_sentiment(self, message: Message) -> Message:
        """Handle a sentiment analysis request."""
        text = message.data.get("text", "")
        if not text:
            return Message(sender=self.agent_id, receiver=message.sender, type="error", data={"error": "No text provided for sentiment analysis"})
        result = self.analyze_sentiment(text)
        return Message(sender=self.agent_id, receiver=message.sender, type="sentiment_results", data={"result": result, "original_text": text})
    
    async def handle_emotion(self, message: Message) -> Message:
        """Handle an emotion tracking request."""
        text = message.data.get("text", "")
        if not text:
            return Message(sender=self.agent_id, receiver=message.sender, type="error", data={"error": "No text provided for emotion tracking"})
        result = self.track_emotions(text)
        return Message(sender=self.agent_id, receiver=message.sender, type="emotion_results", data={"result": result, "original_text": text})
    
    async def handle_pain_points(self, message: Message) -> Message:
        """Handle a pain point identification request."""
        text = message.data.get("text", "")
        if not text:
            return Message(sender=self.agent_id, receiver=message.sender, type="error", data={"error": "No text provided for pain point identification"})
        result = self.identify_pain_points(text)
        return Message(sender=self.agent_id, receiver=message.sender, type="pain_point_results", data={"result": result, "original_text": text})
    
    async def handle_context_analysis(self, message: Message) -> Message:
        """Handle a context-aware analysis request."""
        text = message.data.get("text", "")
        conversation_id = message.data.get("conversation_id", "default")
        if not text:
            return Message(sender=self.agent_id, receiver=message.sender, type="error", data={"error": "No text provided for context analysis"})
        result = self.analyze_with_context(text, conversation_id)
        return Message(sender=self.agent_id, receiver=message.sender, type="context_analysis_results", data={"result": result, "original_text": text})
    
    async def handle_trend_analysis(self, message: Message) -> Message:
        """Handle a trend analysis request."""
        conversation_id = message.data.get("conversation_id", "default")
        trend_type = message.data.get("trend_type", "sentiment")  # sentiment, emotion, pain_point
        emotion_type = message.data.get("emotion_type", None)  # Optional, for emotion trend analysis
        
        if trend_type == "sentiment":
            result = self.analyze_sentiment_trend(conversation_id)
        elif trend_type == "emotion":
            result = self.analyze_emotion_trend(conversation_id, emotion_type)
        elif trend_type == "pain_point":
            result = self.analyze_pain_point_trend(conversation_id)
        else:
            return Message(
                sender=self.agent_id, 
                receiver=message.sender, 
                type="error", 
                data={"error": f"Invalid trend type: {trend_type}. Must be 'sentiment', 'emotion', or 'pain_point'"}
            )
        
        return Message(
            sender=self.agent_id, 
            receiver=message.sender, 
            type="trend_analysis_results", 
            data={
                "result": result, 
                "trend_type": trend_type,
                "conversation_id": conversation_id
            }
        )
    
    async def shutdown(self):
        """Shutdown the agent."""
        logger.info(f"Shutting down Sentiment Analysis Agent: {self.agent_id}")
        # Clear conversation history
        self.conversation_history.clear()
        # Clear trend analysis history
        self.sentiment_history.clear()
        self.emotion_history.clear()
        self.pain_point_history.clear()
        await super().shutdown()
