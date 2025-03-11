"""
NLP Processing Agent for AMPTALK.

This agent provides advanced natural language processing capabilities including:
- Entity detection
- Topic modeling
- Intent recognition
- Language detection
- Contextual understanding

It leverages state-of-the-art transformer models and techniques to provide
high-quality NLP analysis for the AMPTALK multi-agent system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, field

# Import core agent functionality
from src.core.agent import Agent
from src.core.message import Message

# Import model management
from src.models import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class NLPProcessingConfig:
    """Configuration for the NLP Processing Agent."""
    
    # Entity detection
    entity_model_name: str = "dslim/bert-base-NER"
    entity_threshold: float = 0.8
    
    # Topic modeling
    topic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    num_topics: int = 10
    
    # Intent recognition
    intent_model_name: str = "facebook/bart-large-mnli"
    intent_threshold: float = 0.7
    default_intents: List[str] = field(default_factory=lambda: [
        "question", "statement", "request", "command", "greeting", 
        "farewell", "confirmation", "denial", "clarification"
    ])
    
    # Language detection
    language_model_name: str = "papluca/xlm-roberta-base-language-detection"
    
    # Contextual understanding
    context_window_size: int = 5  # Number of previous messages to consider for context
    
    # General settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    cache_dir: str = "model_cache"


class NLPProcessingAgent(Agent):
    """
    NLP Processing Agent that provides advanced natural language processing capabilities.
    
    This agent uses transformer models to perform:
    - Entity detection: Identify and classify named entities in text
    - Topic modeling: Discover abstract topics in text collections
    - Intent recognition: Determine the purpose or goal of a message
    - Language detection: Identify the language of a text
    - Contextual understanding: Analyze text in the context of previous messages
    """
    
    def __init__(
        self,
        agent_id: str,
        config: Optional[NLPProcessingConfig] = None,
        model_manager: Optional[ModelManager] = None
    ):
        """
        Initialize the NLP Processing Agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration for the NLP processing capabilities
            model_manager: Model manager for efficient model loading/unloading
        """
        super().__init__(agent_id=agent_id)
        
        self.config = config or NLPProcessingConfig()
        self.model_manager = model_manager or ModelManager(
            cache_dir=self.config.cache_dir,
            enable_gpu=(self.config.device == "cuda")
        )
        
        # Initialize models as None, they will be loaded on demand
        self._entity_model = None
        self._entity_tokenizer = None
        
        self._topic_model = None
        self._topic_tokenizer = None
        self._topic_clusters = None
        
        self._intent_model = None
        self._intent_tokenizer = None
        
        self._language_model = None
        self._language_tokenizer = None
        
        # Context storage
        self._context_history = {}  # Mapping from conversation ID to list of messages
        
        # Register message handlers
        self.register_handler("process_text", self.handle_process_text)
        self.register_handler("detect_entities", self.handle_detect_entities)
        self.register_handler("model_topics", self.handle_model_topics)
        self.register_handler("recognize_intent", self.handle_recognize_intent)
        self.register_handler("detect_language", self.handle_detect_language)
        self.register_handler("analyze_with_context", self.handle_analyze_with_context)
        
        logger.info(f"NLP Processing Agent initialized with device: {self.config.device}")
        
    async def handle_process_text(self, message: Message) -> Message:
        """
        Process text with all available NLP capabilities.
        
        Args:
            message: Message containing text to process
            
        Returns:
            Message with NLP analysis results
        """
        text = message.data.get("text", "")
        conversation_id = message.data.get("conversation_id", "default")
        
        if not text:
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                type="error",
                data={"error": "No text provided for processing"}
            )
            
        # Update context history
        self._update_context(conversation_id, text)
        
        # Process with all capabilities
        entities = await self.detect_entities(text)
        topics = await self.model_topics(text)
        intent = await self.recognize_intent(text)
        language = await self.detect_language(text)
        context_analysis = await self.analyze_with_context(text, conversation_id)
        
        # Combine results
        results = {
            "entities": entities,
            "topics": topics,
            "intent": intent,
            "language": language,
            "context_analysis": context_analysis,
            "original_text": text
        }
        
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            type="nlp_results",
            data=results
        )
        
    async def handle_detect_entities(self, message: Message) -> Message:
        """Handle entity detection request."""
        text = message.data.get("text", "")
        
        if not text:
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                type="error",
                data={"error": "No text provided for entity detection"}
            )
            
        entities = await self.detect_entities(text)
        
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            type="entity_results",
            data={"entities": entities, "original_text": text}
        )
        
    async def handle_model_topics(self, message: Message) -> Message:
        """Handle topic modeling request."""
        texts = message.data.get("texts", [])
        
        if not texts:
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                type="error",
                data={"error": "No texts provided for topic modeling"}
            )
            
        topics = await self.model_topics(texts)
        
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            type="topic_results",
            data={"topics": topics, "original_texts": texts}
        )
        
    async def handle_recognize_intent(self, message: Message) -> Message:
        """Handle intent recognition request."""
        text = message.data.get("text", "")
        custom_intents = message.data.get("intents", None)
        
        if not text:
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                type="error",
                data={"error": "No text provided for intent recognition"}
            )
            
        intent = await self.recognize_intent(text, custom_intents)
        
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            type="intent_results",
            data={"intent": intent, "original_text": text}
        )
        
    async def handle_detect_language(self, message: Message) -> Message:
        """Handle language detection request."""
        text = message.data.get("text", "")
        
        if not text:
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                type="error",
                data={"error": "No text provided for language detection"}
            )
            
        language = await self.detect_language(text)
        
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            type="language_results",
            data={"language": language, "original_text": text}
        )
        
    async def handle_analyze_with_context(self, message: Message) -> Message:
        """Handle contextual analysis request."""
        text = message.data.get("text", "")
        conversation_id = message.data.get("conversation_id", "default")
        
        if not text:
            return Message(
                sender=self.agent_id,
                receiver=message.sender,
                type="error",
                data={"error": "No text provided for contextual analysis"}
            )
            
        # Update context history
        self._update_context(conversation_id, text)
        
        context_analysis = await self.analyze_with_context(text, conversation_id)
        
        return Message(
            sender=self.agent_id,
            receiver=message.sender,
            type="context_results",
            data={"context_analysis": context_analysis, "original_text": text}
        )
        
    def _update_context(self, conversation_id: str, text: str):
        """Update the context history for a conversation."""
        if conversation_id not in self._context_history:
            self._context_history[conversation_id] = []
            
        self._context_history[conversation_id].append(text)
        
        # Keep only the most recent messages within the context window
        if len(self._context_history[conversation_id]) > self.config.context_window_size:
            self._context_history[conversation_id] = self._context_history[conversation_id][
                -self.config.context_window_size:
            ]
            
    def _get_context(self, conversation_id: str) -> List[str]:
        """Get the context history for a conversation."""
        return self._context_history.get(conversation_id, [])
        
    async def _load_entity_model(self):
        """Load the entity detection model."""
        if self._entity_model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForTokenClassification
                
                logger.info(f"Loading entity detection model: {self.config.entity_model_name}")
                
                self._entity_tokenizer = AutoTokenizer.from_pretrained(self.config.entity_model_name)
                self._entity_model = AutoModelForTokenClassification.from_pretrained(
                    self.config.entity_model_name
                )
                
                # Move model to appropriate device
                self._entity_model = self._entity_model.to(self.config.device)
                
                logger.info("Entity detection model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load entity detection model: {str(e)}")
                raise
                
    async def _load_topic_model(self):
        """Load the topic modeling model."""
        if self._topic_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                from sklearn.feature_extraction.text import CountVectorizer
                from bertopic import BERTopic
                from umap import UMAP
                from hdbscan import HDBSCAN
                
                logger.info(f"Loading topic modeling model: {self.config.topic_model_name}")
                
                # Load sentence transformer for embeddings
                embedding_model = SentenceTransformer(self.config.topic_model_name)
                
                # Configure UMAP for dimensionality reduction
                umap_model = UMAP(
                    n_neighbors=15,
                    n_components=5,
                    min_dist=0.0,
                    metric='cosine',
                    random_state=42
                )
                
                # Configure HDBSCAN for clustering
                hdbscan_model = HDBSCAN(
                    min_cluster_size=15,
                    metric='euclidean',
                    cluster_selection_method='eom',
                    prediction_data=True
                )
                
                # Configure CountVectorizer for topic representation
                vectorizer_model = CountVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2)
                )
                
                # Create BERTopic model
                self._topic_model = BERTopic(
                    embedding_model=embedding_model,
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    vectorizer_model=vectorizer_model,
                    nr_topics=self.config.num_topics
                )
                
                logger.info("Topic modeling model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load topic modeling model: {str(e)}")
                raise
                
    async def _load_intent_model(self):
        """Load the intent recognition model."""
        if self._intent_model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                logger.info(f"Loading intent recognition model: {self.config.intent_model_name}")
                
                self._intent_tokenizer = AutoTokenizer.from_pretrained(self.config.intent_model_name)
                self._intent_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.intent_model_name
                )
                
                # Move model to appropriate device
                self._intent_model = self._intent_model.to(self.config.device)
                
                logger.info("Intent recognition model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load intent recognition model: {str(e)}")
                raise
                
    async def _load_language_model(self):
        """Load the language detection model."""
        if self._language_model is None:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                
                logger.info(f"Loading language detection model: {self.config.language_model_name}")
                
                self._language_tokenizer = AutoTokenizer.from_pretrained(self.config.language_model_name)
                self._language_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.language_model_name
                )
                
                # Move model to appropriate device
                self._language_model = self._language_model.to(self.config.device)
                
                logger.info("Language detection model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load language detection model: {str(e)}")
                raise
                
    async def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect named entities in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities with type, text, and confidence
        """
        await self._load_entity_model()
        
        try:
            # Tokenize input
            inputs = self._entity_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.config.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self._entity_model(**inputs)
                
            # Process outputs
            predictions = torch.argmax(outputs.logits, dim=2)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=2)
            confidence_scores = torch.max(probabilities, dim=2).values
            
            # Convert predictions to entities
            tokens = self._entity_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            token_predictions = [
                (token, self._entity_model.config.id2label[prediction.item()], confidence.item())
                for token, prediction, confidence in zip(tokens, predictions[0], confidence_scores[0])
            ]
            
            # Group tokens into entities
            entities = []
            current_entity = None
            
            for token, label, confidence in token_predictions:
                # Skip special tokens
                if token.startswith("##"):
                    if current_entity:
                        current_entity["text"] += token[2:]
                    continue
                    
                if token in self._entity_tokenizer.all_special_tokens:
                    continue
                    
                # Handle entity labels (typically in BIO format)
                if label.startswith("B-"):
                    # Start a new entity
                    if current_entity:
                        if current_entity["confidence"] >= self.config.entity_threshold:
                            entities.append(current_entity)
                            
                    entity_type = label[2:]  # Remove B- prefix
                    current_entity = {
                        "type": entity_type,
                        "text": token,
                        "confidence": confidence
                    }
                    
                elif label.startswith("I-") and current_entity:
                    # Continue current entity
                    current_entity["text"] += " " + token
                    # Update confidence as average
                    current_entity["confidence"] = (current_entity["confidence"] + confidence) / 2
                    
                else:
                    # End current entity
                    if current_entity:
                        if current_entity["confidence"] >= self.config.entity_threshold:
                            entities.append(current_entity)
                        current_entity = None
                        
            # Add final entity if exists
            if current_entity and current_entity["confidence"] >= self.config.entity_threshold:
                entities.append(current_entity)
                
            return entities
            
        except Exception as e:
            logger.error(f"Error in entity detection: {str(e)}")
            return []
            
    async def model_topics(self, texts: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Perform topic modeling on a collection of texts.
        
        Args:
            texts: Single text or list of texts to analyze
            
        Returns:
            Dictionary with topics and their keywords
        """
        await self._load_topic_model()
        
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
                
            # Fit or transform based on whether we have existing topics
            if self._topic_clusters is None:
                topics, probs = self._topic_model.fit_transform(texts)
                self._topic_clusters = True
            else:
                topics, probs = self._topic_model.transform(texts)
                
            # Get topic info
            topic_info = self._topic_model.get_topic_info()
            
            # Get topic representations
            topic_representations = {}
            for topic_id in set(topics):
                if topic_id != -1:  # Skip outlier topic
                    words = self._topic_model.get_topic(topic_id)
                    topic_representations[str(topic_id)] = {
                        "keywords": [word for word, _ in words],
                        "weights": [float(weight) for _, weight in words]
                    }
                    
            # Get document-topic mapping
            document_topics = []
            for i, (doc_topic, doc_prob) in enumerate(zip(topics, probs)):
                if doc_topic != -1:  # Skip outlier topic
                    document_topics.append({
                        "document_id": i,
                        "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                        "topic_id": int(doc_topic),
                        "probability": float(doc_prob)
                    })
                    
            return {
                "topic_representations": topic_representations,
                "document_topics": document_topics,
                "topic_info": topic_info.to_dict("records")
            }
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {str(e)}")
            return {"error": str(e)}
            
    async def recognize_intent(
        self, 
        text: str, 
        custom_intents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Recognize the intent of a text.
        
        Args:
            text: Text to analyze
            custom_intents: Optional list of custom intents to check
            
        Returns:
            Dictionary with recognized intent and confidence
        """
        await self._load_intent_model()
        
        try:
            # Use default or custom intents
            intents = custom_intents if custom_intents else self.config.default_intents
            
            # Prepare hypothesis template for zero-shot classification
            hypothesis_template = "This text is about {}."
            
            # Create hypothesis for each intent
            candidate_labels = intents
            hypothesis = [hypothesis_template.format(intent) for intent in candidate_labels]
            
            # Tokenize premise and hypothesis
            with torch.no_grad():
                # Process each hypothesis against the text (premise)
                scores = []
                for hyp in hypothesis:
                    inputs = self._intent_tokenizer(
                        text,
                        hyp,
                        return_tensors="pt",
                        truncation=True,
                        padding=True
                    ).to(self.config.device)
                    
                    outputs = self._intent_model(**inputs)
                    prediction = torch.nn.functional.softmax(outputs.logits, dim=1)
                    # For MNLI model, index 2 corresponds to "entailment"
                    entailment_score = prediction[0, 2].item()
                    scores.append(entailment_score)
                    
            # Find the highest scoring intent
            max_score_index = np.argmax(scores)
            max_score = scores[max_score_index]
            
            # Only return intent if confidence is above threshold
            if max_score >= self.config.intent_threshold:
                intent = {
                    "intent": candidate_labels[max_score_index],
                    "confidence": max_score,
                    "all_intents": [
                        {"intent": intent, "confidence": score}
                        for intent, score in zip(candidate_labels, scores)
                    ]
                }
            else:
                intent = {
                    "intent": "unknown",
                    "confidence": 1.0 - max_score,
                    "all_intents": [
                        {"intent": intent, "confidence": score}
                        for intent, score in zip(candidate_labels, scores)
                    ]
                }
                
            return intent
            
        except Exception as e:
            logger.error(f"Error in intent recognition: {str(e)}")
            return {"intent": "error", "confidence": 0.0, "error": str(e)}
            
    async def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of a text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with detected language and confidence
        """
        await self._load_language_model()
        
        try:
            # Tokenize input
            inputs = self._language_tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.config.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self._language_model(**inputs)
                
            # Process outputs
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            confidence, predicted_class = torch.max(probabilities, dim=0)
            
            # Get language label
            language_id = predicted_class.item()
            language = self._language_model.config.id2label[language_id]
            
            return {
                "language": language,
                "confidence": confidence.item(),
                "language_code": language.split('_')[0] if '_' in language else language
            }
            
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            return {"language": "unknown", "confidence": 0.0, "error": str(e)}
            
    async def analyze_with_context(self, text: str, conversation_id: str) -> Dict[str, Any]:
        """
        Analyze text with context from previous messages.
        
        Args:
            text: Text to analyze
            conversation_id: ID of the conversation for context
            
        Returns:
            Dictionary with contextual analysis results
        """
        # Get context history
        context = self._get_context(conversation_id)
        
        if len(context) <= 1:
            # Not enough context for analysis
            return {
                "has_context": False,
                "context_size": len(context),
                "analysis": "Insufficient context for analysis"
            }
            
        try:
            # Combine context into a single text for analysis
            context_text = " ".join(context[:-1])  # Exclude current message
            
            # Detect entities in context and current message
            context_entities = await self.detect_entities(context_text)
            current_entities = await self.detect_entities(text)
            
            # Find references to previous entities
            referenced_entities = []
            for entity in current_entities:
                for context_entity in context_entities:
                    if entity["type"] == context_entity["type"] or entity["text"].lower() in context_entity["text"].lower():
                        referenced_entities.append({
                            "current": entity,
                            "referenced": context_entity
                        })
                        
            # Detect if current message is a follow-up question or response
            is_followup = False
            followup_confidence = 0.0
            
            # Check for question words without subject
            question_starters = ["what", "when", "where", "which", "who", "whom", "whose", "why", "how"]
            first_word = text.strip().split()[0].lower() if text.strip() else ""
            
            if first_word in question_starters:
                is_followup = True
                followup_confidence = 0.8
                
            # Check for pronouns referring to previous context
            pronouns = ["it", "this", "that", "these", "those", "they", "them", "he", "she", "his", "her"]
            words = text.lower().split()
            
            for pronoun in pronouns:
                if pronoun in words:
                    is_followup = True
                    followup_confidence = max(followup_confidence, 0.7)
                    
            return {
                "has_context": True,
                "context_size": len(context),
                "referenced_entities": referenced_entities,
                "is_followup": is_followup,
                "followup_confidence": followup_confidence,
                "context_summary": f"Context of {len(context)-1} previous messages"
            }
            
        except Exception as e:
            logger.error(f"Error in contextual analysis: {str(e)}")
            return {
                "has_context": True,
                "context_size": len(context),
                "error": str(e)
            }
            
    async def shutdown(self):
        """Clean up resources when shutting down the agent."""
        logger.info(f"Shutting down NLP Processing Agent: {self.agent_id}")
        
        # Clear models to free memory
        self._entity_model = None
        self._entity_tokenizer = None
        
        self._topic_model = None
        self._topic_tokenizer = None
        self._topic_clusters = None
        
        self._intent_model = None
        self._intent_tokenizer = None
        
        self._language_model = None
        self._language_tokenizer = None
        
        # Clear context history
        self._context_history = {}
        
        await super().shutdown() 