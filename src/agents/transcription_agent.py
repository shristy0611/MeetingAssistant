"""
Transcription Agent for AMPTALK Framework.

This module defines a specialized agent for audio transcription
using the Whisper model.

Author: AMPTALK Team
Date: 2024
"""

import logging
import asyncio
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import traceback
import tempfile

from src.core.framework.agent import Agent
from src.core.framework.message import (
    Message, MessageType, MessagePriority,
    create_transcription_result_message
)
from src.core.utils.logging_config import get_logger
from src.core.utils.model_cache import get_model_cache
try:
    from src.core.utils.edge_optimization import (
        EdgeOptimizer, OptimizationLevel, DeviceTarget, 
        optimize_whisper_model, OptimizationType
    )
    EDGE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    EDGE_OPTIMIZATION_AVAILABLE = False

# Configure logger
logger = get_logger("amptalk.agents.transcription")

# Import optional dependencies
try:
    import psutil
except ImportError:
    psutil = None


class ModelType:
    """Type of model to use for transcription."""
    WHISPER_TRANSFORMERS = "whisper_transformers"  # Use transformers library
    WHISPER_FASTER = "whisper_faster"  # Use faster_whisper (CTranslate2)
    WHISPER_OPTIMIZED = "whisper_optimized"  # Use optimized (ONNX) version
    WHISPER_OPTIMIZED_ALL_IN_ONE = "whisper_optimized_all_in_one"  # All-in-one ONNX model
    WHISPER_TFLITE = "whisper_tflite"  # TensorFlow Lite model
    WHISPER_COREML = "whisper_coreml"  # Apple Core ML model


class WhisperModelSize:
    """Standard Whisper model sizes."""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    LARGE_V3 = "large-v3"


class WhisperModelConfig:
    """Configuration for the Whisper transcription model."""
    
    def __init__(self,
                model_size: str = "large-v3-turbo",
                model_path: Optional[str] = None,
                language: Optional[str] = None,
                compute_type: str = "auto",
                device: str = "auto",
                beam_size: int = 5,
                best_of: int = 5,
                patience: float = 1.0,
                length_penalty: float = 1.0,
                temperature: float = 0.0,
                compression_ratio_threshold: float = 2.4,
                log_prob_threshold: float = -1.0,
                no_speech_threshold: float = 0.6,
                condition_on_previous_text: bool = True,
                initial_prompt: Optional[str] = None,
                word_timestamps: bool = True):
        """
        Initialize Whisper model configuration.
        
        Args:
            model_size: Size of the Whisper model ("tiny", "base", "small", "medium", "large-v3-turbo", etc.)
            model_path: Optional path to a local model file
            language: Optional language code for transcription (e.g., "en", "fr")
            compute_type: Computation precision ("float16", "int8", "auto")
            device: Device to run inference on ("cpu", "cuda", "mps", "auto")
            beam_size: Beam size for beam search
            best_of: Number of candidates when sampling with non-zero temperature
            patience: Beam search patience factor
            length_penalty: Exponential length penalty constant
            temperature: Temperature for sampling
            compression_ratio_threshold: If the compression ratio exceeds this value, treat as failed
            log_prob_threshold: If the average log probability is below this value, treat as failed
            no_speech_threshold: If the no_speech probability exceeds this value, treat as silence
            condition_on_previous_text: Whether to condition on previous text
            initial_prompt: Optional text to provide as a prompt for the first window
            word_timestamps: Whether to include word-level timestamps
        """
        self.model_size = model_size
        self.model_path = model_path
        self.language = language
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.best_of = best_of
        self.patience = patience
        self.length_penalty = length_penalty
        self.temperature = temperature
        self.compression_ratio_threshold = compression_ratio_threshold
        self.log_prob_threshold = log_prob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.condition_on_previous_text = condition_on_previous_text
        self.initial_prompt = initial_prompt
        self.word_timestamps = word_timestamps
        
        # Cache for model instances
        self.model_instance = None


class TranscriptionAgent(Agent):
    """
    Agent specialized for audio transcription using Whisper.
    
    This agent can transcribe audio files to text using various
    backends of the Whisper model, including optimized versions
    for edge devices.
    """
    
    def __init__(self, 
                 agent_id: str,
                 model_size: str = WhisperModelSize.TINY,
                 model_type: str = ModelType.WHISPER_TRANSFORMERS,
                 language: Optional[str] = None,
                 use_model_caching: bool = True,
                 optimization_level: Optional[str] = None,
                 target_device: str = "cpu",
                 model_path: Optional[str] = None,  # Custom model path for TFLite/CoreML
                 **kwargs):
        """
        Initialize a transcription agent.
        
        Args:
            agent_id: Unique identifier for the agent
            model_size: Size of the Whisper model to use
            model_type: Type of model implementation to use
            language: Optional language code for the model
            use_model_caching: Whether to cache models between calls
            optimization_level: Level of optimization for edge deployment
            target_device: Target device for optimized models
            model_path: Custom path to pre-optimized model files (TFLite/CoreML)
            **kwargs: Additional arguments to pass to the Agent constructor
        """
        super().__init__(agent_id=agent_id, **kwargs)
        
        self.model_size = model_size
        self.model_type = model_type
        self.language = language
        self.use_model_caching = use_model_caching
        self.model_path = model_path
        
        # Set up optimization configuration
        self.optimization_level = optimization_level
        self.target_device = target_device
        self._edge_optimizer = None
        
        # Set up model caching
        self._model = None
        self._processor = None
        self._model_last_used = 0
        self._model_cache_ttl = 300  # 5 minutes
        
        # Check required dependencies based on model type
        self._check_dependencies()
        
        logger.info(f"Initialized TranscriptionAgent with {model_size} model "
                   f"using {model_type} backend")
    
    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            if self.model_type == ModelType.WHISPER_TRANSFORMERS:
                try:
                    from transformers import WhisperProcessor, WhisperForConditionalGeneration
                    self._has_transformers = True
                except ImportError:
                    self._has_transformers = False
                    logger.warning("Could not import transformers, WhisperTransformers will not be available")
                
            elif self.model_type == ModelType.WHISPER_FASTER:
                try:
                    from faster_whisper import WhisperModel
                    self._has_faster_whisper = True
                except ImportError:
                    self._has_faster_whisper = False
                    logger.warning("Could not import faster_whisper, WhisperFaster will not be available")
                    
            elif self.model_type == ModelType.WHISPER_OPTIMIZED:
                if not EDGE_OPTIMIZATION_AVAILABLE:
                    logger.warning("Edge optimization utilities not available, falling back to standard model")
                    self.model_type = ModelType.WHISPER_TRANSFORMERS
                else:
                    try:
                        import onnxruntime
                        self._has_onnxruntime = True
                    except ImportError:
                        self._has_onnxruntime = False
                        logger.warning("Could not import onnxruntime, optimized models will not be available")
                        self.model_type = ModelType.WHISPER_TRANSFORMERS
                        
            elif self.model_type == ModelType.WHISPER_TFLITE:
                try:
                    import tensorflow as tf
                    self._has_tensorflow = True
                    if not hasattr(tf, 'lite'):
                        logger.warning("TensorFlow Lite not available in this TensorFlow installation")
                        self._has_tflite = False
                    else:
                        self._has_tflite = True
                except ImportError:
                    self._has_tensorflow = False
                    self._has_tflite = False
                    logger.warning("TensorFlow not available, TFLite models will not work")
            
            elif self.model_type == ModelType.WHISPER_COREML:
                try:
                    # Only relevant on macOS/iOS, so we just check if we're on the right platform
                    import platform
                    if platform.system() != "Darwin":
                        logger.warning("Core ML is only available on macOS/iOS")
                        self._has_coreml = False
                    else:
                        # Try importing coremltools for validation
                        try:
                            import coremltools
                            self._has_coreml = True
                        except ImportError:
                            self._has_coreml = False
                            logger.warning("coremltools not available for validation")
                except Exception as e:
                    self._has_coreml = False
                    logger.warning(f"Error checking Core ML availability: {e}")
            
            # Check for audio processing dependencies
            try:
                import librosa
                self._has_librosa = True
            except ImportError:
                self._has_librosa = False
                logger.warning("Could not import librosa, audio preprocessing may be limited")
                
        except Exception as e:
            logger.error(f"Error checking dependencies: {e}")
    
    async def _initialize_model(self):
        """Initialize the model if not already loaded."""
        # Skip if model is already loaded and caching is enabled
        if self._model is not None and self.use_model_caching:
            self._model_last_used = time.time()
            return
            
        try:
            # Initialize based on model type
            if self.model_type == ModelType.WHISPER_TRANSFORMERS:
                await self._initialize_transformers_model()
                
            elif self.model_type == ModelType.WHISPER_FASTER:
                await self._initialize_faster_model()
                
            elif self.model_type == ModelType.WHISPER_OPTIMIZED:
                await self._initialize_optimized_model()
                
            elif self.model_type == ModelType.WHISPER_TFLITE:
                await self._initialize_tflite_model()
                
            elif self.model_type == ModelType.WHISPER_COREML:
                await self._initialize_coreml_model()
                
            # Update last used timestamp
            self._model_last_used = time.time()
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise RuntimeError(f"Failed to initialize transcription model: {e}")
    
    async def _initialize_transformers_model(self):
        """Initialize the Whisper model using the transformers library."""
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        logger.info(f"Initializing Whisper model (transformers) with size {self.model_size}")
        
        # Determine model ID
        lang_suffix = f".{self.language}" if self.language else ""
        model_id = f"openai/whisper-{self.model_size}{lang_suffix}"
        
        # Load model asynchronously
        loop = asyncio.get_event_loop()
        
        # Load processor
        self._processor = await loop.run_in_executor(
            None, 
            lambda: WhisperProcessor.from_pretrained(model_id)
        )
        
        # Load model
        self._model = await loop.run_in_executor(
            None, 
            lambda: WhisperForConditionalGeneration.from_pretrained(model_id)
        )
        
        logger.info(f"Whisper model (transformers) initialized successfully")
    
    async def _initialize_faster_model(self):
        """Initialize the Whisper model using the faster_whisper library."""
        from faster_whisper import WhisperModel
        
        logger.info(f"Initializing Whisper model (faster) with size {self.model_size}")
        
        # Load model asynchronously
        loop = asyncio.get_event_loop()
        
        # Determine compute type based on available hardware
        compute_type = "float16"  # Default to float16
        
        # Load model
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self.model_size,
                device=self.target_device,
                compute_type=compute_type,
                language=self.language
            )
        )
        
        # No separate processor for faster_whisper
        self._processor = None
        
        logger.info(f"Whisper model (faster) initialized successfully")
    
    async def _initialize_optimized_model(self):
        """Initialize the optimized Whisper model for edge deployment."""
        if not EDGE_OPTIMIZATION_AVAILABLE:
            logger.warning("Edge optimization not available, falling back to transformers")
            return await self._initialize_transformers_model()
            
        logger.info(f"Initializing optimized Whisper model with size {self.model_size}")
        
        # Create edge optimizer if not already created
        if self._edge_optimizer is None:
            # Convert string optimization level to enum
            if self.optimization_level:
                opt_level = getattr(OptimizationLevel, self.optimization_level.upper(), 
                                   OptimizationLevel.MEDIUM)
            else:
                opt_level = OptimizationLevel.MEDIUM
                
            # Convert string target device to enum
            target_dev = getattr(DeviceTarget, self.target_device.upper(), 
                                DeviceTarget.CPU)
                
            self._edge_optimizer = EdgeOptimizer(
                optimization_level=opt_level,
                target_device=target_dev,
                cache_dir=os.path.join(os.getcwd(), "models", "optimized")
            )
        
        # Determine if we should use all-in-one model
        all_in_one = self.model_type == ModelType.WHISPER_OPTIMIZED_ALL_IN_ONE
        
        # Get or create optimized model
        optimization_result = self._edge_optimizer.optimize_whisper(
            model_size=self.model_size,
            language=self.language,
            all_in_one=all_in_one
        )
        
        if "error" in optimization_result:
            logger.error(f"Error optimizing model: {optimization_result['error']}")
            logger.warning("Falling back to standard transformers model")
            return await self._initialize_transformers_model()
            
        # Load the optimized model
        model_path = optimization_result["model_path"]
        logger.info(f"Loading optimized model from {model_path}")
        
        # Store the optimization results for later use
        self._optimization_result = optimization_result
        
        # Check if this is an all-in-one model
        if all_in_one:
            await self._initialize_all_in_one_onnx_model(model_path)
        else:
            # For separate encoder/decoder models
            await self._initialize_separate_onnx_models(model_path)
    
    async def _initialize_all_in_one_onnx_model(self, model_path: str):
        """
        Initialize an all-in-one ONNX model that combines encoder, decoder and beam search.
        
        Args:
            model_path: Path to the all-in-one ONNX model
        """
        import onnxruntime as ort
        from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
        
        logger.info(f"Initializing all-in-one ONNX model from {model_path}")
        
        # Check if we need to register custom ops
        try:
            # Try to import onnxruntime extensions
            from onnxruntime_extensions import get_library_path
            
            # Create session options with custom ops
            session_options = ort.SessionOptions()
            session_options.register_custom_ops_library(get_library_path())
            
            # Create inference session
            self._model = ort.InferenceSession(
                model_path, 
                sess_options=session_options,
                providers=["CPUExecutionProvider"]
            )
            
        except ImportError:
            # If extensions not available, try without them
            logger.warning("onnxruntime_extensions not found, trying without custom ops")
            
            try:
                # Create session without custom ops
                session_options = ort.SessionOptions()
                self._model = ort.InferenceSession(
                    model_path, 
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"]
                )
            except InvalidArgument as e:
                logger.error(f"Failed to initialize all-in-one model without extensions: {e}")
                logger.error("Your all-in-one model may require onnxruntime_extensions")
                raise RuntimeError("Failed to initialize all-in-one ONNX model") from e
        
        # Get model inputs/outputs
        model_inputs = self._model.get_inputs()
        model_outputs = self._model.get_outputs()
        
        logger.info(f"ONNX model loaded with {len(model_inputs)} inputs and {len(model_outputs)} outputs")
        
        # For all-in-one models, we may still need the processor for tokenization
        from transformers import WhisperProcessor
        lang_suffix = f".{self.language}" if self.language else ""
        model_id = f"openai/whisper-{self.model_size}{lang_suffix}"
        
        # Load processor asynchronously
        loop = asyncio.get_event_loop()
        self._processor = await loop.run_in_executor(
            None, 
            lambda: WhisperProcessor.from_pretrained(model_id)
        )
        
        # Set the model type
        self._onnx_model_type = "all_in_one"
        logger.info("All-in-one ONNX model initialized successfully")
    
    async def _initialize_separate_onnx_models(self, encoder_path: str):
        """
        Initialize separate ONNX encoder and decoder models.
        
        Args:
            encoder_path: Path to the ONNX encoder model
        """
        import onnxruntime as ort
        
        logger.info(f"Initializing separate ONNX models from {encoder_path}")
        
        # Assume the decoder model is in the same directory
        model_dir = os.path.dirname(encoder_path)
        
        # Look for the decoder models
        decoder_path = None
        decoder_with_kv_path = None
        
        for file in os.listdir(model_dir):
            if file.endswith(".onnx"):
                if "decoder" in file and "kv" not in file:
                    decoder_path = os.path.join(model_dir, file)
                elif "decoder" in file and "kv" in file:
                    decoder_with_kv_path = os.path.join(model_dir, file)
        
        # Check if we found decoder models
        if not decoder_path:
            logger.warning("No decoder model found, only encoder will be available")
        
        # Create session options
        session_options = ort.SessionOptions()
        
        # Initialize encoder model
        self._encoder_model = ort.InferenceSession(
            encoder_path, 
            sess_options=session_options,
            providers=["CPUExecutionProvider"]
        )
        
        # Initialize decoder model if available
        self._decoder_model = None
        if decoder_path:
            self._decoder_model = ort.InferenceSession(
                decoder_path, 
                sess_options=session_options,
                providers=["CPUExecutionProvider"]
            )
        
        # Initialize decoder with KV caching if available
        self._decoder_with_kv_model = None
        if decoder_with_kv_path:
            self._decoder_with_kv_model = ort.InferenceSession(
                decoder_with_kv_path, 
                sess_options=session_options,
                providers=["CPUExecutionProvider"]
            )
        
        # For ONNX models, we need the processor for tokenization
        from transformers import WhisperProcessor
        lang_suffix = f".{self.language}" if self.language else ""
        model_id = f"openai/whisper-{self.model_size}{lang_suffix}"
        
        # Load processor asynchronously
        loop = asyncio.get_event_loop()
        self._processor = await loop.run_in_executor(
            None, 
            lambda: WhisperProcessor.from_pretrained(model_id)
        )
        
        # Get input/output names
        self._encoder_input_name = self._encoder_model.get_inputs()[0].name
        self._encoder_output_name = self._encoder_model.get_outputs()[0].name
        
        if self._decoder_model:
            self._decoder_input_names = [input.name for input in self._decoder_model.get_inputs()]
            self._decoder_output_names = [output.name for output in self._decoder_model.get_outputs()]
        
        # Set the model type
        self._onnx_model_type = "separate"
        logger.info("Separate ONNX models initialized successfully")
    
    async def _unload_model_if_inactive(self):
        """Unload the model if it hasn't been used for a while."""
        if not self.use_model_caching and self._model is not None:
            # Check if model has been inactive for too long
            if time.time() - self._model_last_used > self._model_cache_ttl:
                logger.info(f"Unloading inactive model (TTL: {self._model_cache_ttl}s)")
                
                # Clear model and processor
                self._model = None
                self._processor = None
    
    async def _preprocess_audio(self, audio_file_path: str) -> Any:
        """
        Preprocess audio for transcription.
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Preprocessed audio data ready for the model
        """
        try:
            if self.model_type == ModelType.WHISPER_TRANSFORMERS:
                import librosa
                import numpy as np
                import torch
                
                # Load audio asynchronously
                loop = asyncio.get_event_loop()
                audio_array, sampling_rate = await loop.run_in_executor(
                    None,
                    lambda: librosa.load(audio_file_path, sr=16000)
                )
                
                # Convert to expected format
                inputs = self._processor(
                    audio_array, 
                    sampling_rate=sampling_rate, 
                    return_tensors="pt"
                )
                
                return inputs
                
            elif self.model_type == ModelType.WHISPER_FASTER:
                # faster_whisper handles audio loading internally
                return audio_file_path
                
            elif self.model_type == ModelType.WHISPER_OPTIMIZED:
                # For ONNX models
                if hasattr(self._model, "run"):  # ONNX Runtime session
                    import librosa
                    import numpy as np
                    
                    # Load audio asynchronously
                    loop = asyncio.get_event_loop()
                    audio_array, sampling_rate = await loop.run_in_executor(
                        None,
                        lambda: librosa.load(audio_file_path, sr=16000)
                    )
                    
                    # Get feature extractor
                    feature_extractor = self._processor.feature_extractor
                    inputs = feature_extractor(
                        audio_array, 
                        sampling_rate=sampling_rate, 
                        return_tensors="np"
                    )
                    
                    return inputs.input_features
                else:
                    # Fall back to transformers preprocessing
                    import librosa
                    import numpy as np
                    import torch
                    
                    # Load audio asynchronously
                    loop = asyncio.get_event_loop()
                    audio_array, sampling_rate = await loop.run_in_executor(
                        None,
                        lambda: librosa.load(audio_file_path, sr=16000)
                    )
                    
                    # Convert to expected format
                    inputs = self._processor(
                        audio_array, 
                        sampling_rate=sampling_rate, 
                        return_tensors="pt"
                    )
                    
                    return inputs
                
        except Exception as e:
            logger.error(f"Error preprocessing audio: {e}")
            raise RuntimeError(f"Failed to preprocess audio: {e}")
    
    async def _transcribe_with_transformers(self, inputs: Any) -> str:
        """
        Transcribe audio using the transformers Whisper model.
        
        Args:
            inputs: Preprocessed audio inputs
            
        Returns:
            Transcription text
        """
        try:
            import torch
            
            loop = asyncio.get_event_loop()
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = await loop.run_in_executor(
                    None,
                    lambda: self._model.generate(inputs.input_features)
                )
                
            # Decode the generated IDs
            transcription = await loop.run_in_executor(
                None,
                lambda: self._processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
            )
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing with transformers: {e}")
            raise RuntimeError(f"Failed to transcribe with transformers: {e}")
    
    async def _transcribe_with_faster(self, audio_path: str) -> str:
        """
        Transcribe audio using the faster_whisper model.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcription text
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Generate transcription
            segments, info = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    audio_path,
                    language=self.language,
                    task="transcribe"
                )
            )
            
            # Collect all segments
            result = ""
            for segment in segments:
                result += segment.text + " "
                
            return result.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing with faster_whisper: {e}")
            raise RuntimeError(f"Failed to transcribe with faster_whisper: {e}")
    
    async def _transcribe_with_optimized(self, inputs: Any) -> str:
        """
        Transcribe audio using the optimized model.
        
        Args:
            inputs: Preprocessed audio inputs
            
        Returns:
            Transcription text
        """
        try:
            if self.model_type == ModelType.WHISPER_OPTIMIZED or self.model_type == ModelType.WHISPER_OPTIMIZED_ALL_IN_ONE:
                if hasattr(self, '_onnx_model_type'):
                    if self._onnx_model_type == "all_in_one":
                        return await self._transcribe_with_all_in_one_onnx(inputs)
                    elif self._onnx_model_type == "separate":
                        return await self._transcribe_with_separate_onnx(inputs)
                
                # Fallback if model type not set
                logger.warning("ONNX model type not determined, using basic inference")
                return await self._transcribe_with_basic_onnx(inputs)
            
            elif self.model_type == ModelType.WHISPER_TFLITE:
                return await self._transcribe_with_tflite(inputs)
            
            elif self.model_type == ModelType.WHISPER_COREML:
                return await self._transcribe_with_coreml(inputs)
            
            else:
                logger.warning(f"Unsupported optimized model type: {self.model_type}")
                return await self._transcribe_with_transformers(inputs)
                
        except Exception as e:
            logger.error(f"Error transcribing with optimized model: {e}")
            raise RuntimeError(f"Failed to transcribe with optimized model: {e}")
    
    async def _transcribe_with_all_in_one_onnx(self, inputs: Any) -> str:
        """
        Transcribe audio using an all-in-one ONNX model.
        
        Args:
            inputs: Preprocessed audio inputs (could be audio file bytes or features)
            
        Returns:
            Transcription text
        """
        import numpy as np
        
        logger.info("Transcribing with all-in-one ONNX model")
        
        # Get all input names
        input_names = [input.name for input in self._model.get_inputs()]
        
        # Check if model expects audio_stream or preprocessed features
        if "audio_stream" in input_names:
            # Model expects raw audio bytes
            if isinstance(inputs, str):
                # If input is a file path, read it
                with open(inputs, "rb") as f:
                    audio_data = np.asarray(list(f.read()), dtype=np.uint8)
            elif isinstance(inputs, np.ndarray):
                # If input is already a numpy array, use it
                audio_data = inputs
            else:
                # If input is from feature extractor, we need to convert it
                raise ValueError("All-in-one model expects audio bytes but received features")
            
            # Prepare inputs for the model
            onnx_inputs = {
                "audio_stream": np.array([audio_data]),
                "max_length": np.array([448], dtype=np.int32),  # Max length for Whisper
                "min_length": np.array([1], dtype=np.int32),
                "num_beams": np.array([5], dtype=np.int32),
                "num_return_sequences": np.array([1], dtype=np.int32),
                "length_penalty": np.array([1.0], dtype=np.float32),
                "repetition_penalty": np.array([1.0], dtype=np.float32),
            }
            
            # Add attention mask if needed
            if "attention_mask" in input_names:
                onnx_inputs["attention_mask"] = np.zeros((1, 80, 3000), dtype=np.int32)
            
        else:
            # Model expects preprocessed features
            if isinstance(inputs, np.ndarray):
                # If input is already a numpy array, use it
                features = inputs
            elif hasattr(inputs, "input_features"):
                # If input is from feature extractor, extract the features
                features = inputs.input_features.numpy()
            else:
                raise ValueError("Model expects preprocessed features")
            
            # Prepare inputs for the model (simplified - actual inputs depend on model)
            onnx_inputs = {
                "input_features": features,
            }
        
        # Filter inputs to only include those expected by the model
        filtered_inputs = {k: v for k, v in onnx_inputs.items() if k in input_names}
        
        # Run inference
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._model.run(None, filtered_inputs)
        )
        
        # Process outputs
        # Usually the output is directly the text or token IDs
        if isinstance(outputs[0], np.ndarray) and outputs[0].dtype == np.int64:
            # If output is token IDs, decode them
            token_ids = outputs[0]
            
            # Decode the token IDs
            transcription = await loop.run_in_executor(
                None,
                lambda: self._processor.batch_decode(token_ids, skip_special_tokens=True)[0]
            )
            
        elif isinstance(outputs[0], np.ndarray) and outputs[0].dtype == np.dtype('S1'):
            # If output is bytes, decode to string
            try:
                transcription = outputs[0][0].decode('utf-8')
            except:
                transcription = str(outputs[0])
        else:
            # If we can't interpret the output, return a placeholder
            transcription = f"ONNX output not interpretable: {type(outputs[0])}"
            logger.warning(transcription)
        
        return transcription
    
    async def _transcribe_with_separate_onnx(self, inputs: Any) -> str:
        """
        Transcribe audio using separate ONNX encoder and decoder models.
        
        Args:
            inputs: Preprocessed audio inputs
            
        Returns:
            Transcription text
        """
        import numpy as np
        import torch
        
        logger.info("Transcribing with separate ONNX models")
        
        # Get tokenizer from processor
        tokenizer = self._processor.tokenizer
        
        # Run encoder
        if hasattr(inputs, "input_features"):
            # If input is from feature extractor, extract the features
            encoder_input = inputs.input_features.numpy()
        else:
            # If input is already a numpy array, use it
            encoder_input = inputs
        
        # Run encoder inference
        loop = asyncio.get_event_loop()
        encoder_output = await loop.run_in_executor(
            None,
            lambda: self._encoder_model.run(
                [self._encoder_output_name], 
                {self._encoder_input_name: encoder_input}
            )[0]
        )
        
        # Check if decoder is available
        if not self._decoder_model:
            logger.warning("No decoder model available, returning partial result")
            return "Encoder-only transcription (not complete)"
        
        # Start autoregressive decoding
        # Get the special tokens
        sot_token = tokenizer.sot
        eot_token = tokenizer.eot
        sot_sequence = tokenizer.sot_sequence_including_notimestamps
        
        # Initialize with the start of text sequence
        tokens = list(sot_sequence)
        
        # Prepare for KV caching if available
        use_kv_caching = self._decoder_with_kv_model is not None
        past_key_values = None
        
        # Decode autoregressively until EOS or max length
        max_length = 448  # Maximum length for Whisper
        
        for i in range(max_length):
            # Convert tokens to tensor
            input_ids = np.array([tokens], dtype=np.int64)
            
            if use_kv_caching and past_key_values is not None:
                # Use KV caching model
                # This requires careful handling of past_key_values
                
                # Create the inputs dict for the KV caching model
                decoder_inputs = {
                    "input_ids": input_ids[:, -1:],  # Only the last token
                    "encoder_hidden_states": encoder_output
                }
                
                # Add past key values to inputs
                for layer_idx, layer_past in enumerate(past_key_values):
                    for state_idx, state in enumerate(layer_past):
                        state_names = ["past_key", "past_value", "past_cross_key", "past_cross_value"]
                        decoder_inputs[f"{state_names[state_idx]}_{layer_idx}"] = state
                
                # Run decoder with KV caching
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self._decoder_with_kv_model.run(None, decoder_inputs)
                )
                
                # Extract logits (should be first output)
                logits = outputs[0]
                
                # Extract updated past key values
                num_layers = len(past_key_values)
                past_key_values = []
                
                for layer_idx in range(num_layers):
                    # Each layer has 4 states
                    layer_past = []
                    for state_idx in range(4):
                        state_names = ["new_key", "new_value", "new_cross_key", "new_cross_value"]
                        output_idx = 1 + layer_idx * 4 + state_idx
                        if output_idx < len(outputs):
                            layer_past.append(outputs[output_idx])
                        else:
                            # If output not found, use a dummy state
                            logger.warning(f"Missing state output at index {output_idx}")
                            layer_past.append(np.zeros((1, 1, 1, 1)))
                    
                    past_key_values.append(tuple(layer_past))
                
            else:
                # Use standard decoder without KV caching
                decoder_inputs = {
                    "input_ids": input_ids,
                    "encoder_hidden_states": encoder_output
                }
                
                # Run standard decoder
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self._decoder_model.run(["logits"], decoder_inputs)
                )
                
                logits = outputs[0]
            
            # Get the next token (greedy decoding)
            next_token_id = int(np.argmax(logits[0, -1, :]))
            
            # Add the token to the sequence
            tokens.append(next_token_id)
            
            # Check if we've reached the end of text token
            if next_token_id == eot_token:
                break
        
        # Decode the tokens
        transcription = await loop.run_in_executor(
            None,
            lambda: tokenizer.decode(tokens, skip_special_tokens=True)
        )
        
        return transcription
    
    async def _transcribe_with_basic_onnx(self, inputs: Any) -> str:
        """
        Transcribe audio using basic ONNX model inference without KV caching.
        
        Args:
            inputs: Preprocessed audio inputs
            
        Returns:
            Transcription text
        """
        # Get input name
        input_name = self._model.get_inputs()[0].name
        
        # Run inference
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: self._model.run(None, {input_name: inputs})
        )
        
        # Since this is a simplified implementation, return a placeholder
        return "Basic ONNX transcription (encoder-only)"
    
    async def _initialize_tflite_model(self):
        """Initialize a TensorFlow Lite model for transcription."""
        if not hasattr(self, '_has_tflite') or not self._has_tflite:
            logger.warning("TensorFlow Lite not available, falling back to transformers")
            return await self._initialize_transformers_model()
        
        logger.info(f"Initializing TensorFlow Lite model")
        
        import tensorflow as tf
        
        # Determine the model path
        if self.model_path:
            # Use provided model path
            tflite_model_path = self.model_path
        else:
            # Use EdgeOptimizer to create or get the model
            if self._edge_optimizer is None:
                # Convert string optimization level to enum
                if self.optimization_level:
                    from src.core.utils.edge_optimization import OptimizationLevel, DeviceTarget, MobileFramework
                    opt_level = getattr(OptimizationLevel, self.optimization_level.upper(), 
                                       OptimizationLevel.MEDIUM)
                else:
                    from src.core.utils.edge_optimization import OptimizationLevel, DeviceTarget, MobileFramework
                    opt_level = OptimizationLevel.MEDIUM
                    
                # Convert string target device to enum
                target_dev = getattr(DeviceTarget, self.target_device.upper(), 
                                    DeviceTarget.CPU)
                    
                self._edge_optimizer = EdgeOptimizer(
                    optimization_level=opt_level,
                    target_device=target_dev,
                    cache_dir=os.path.join(os.getcwd(), "models", "optimized")
                )
            
            # Create or get TFLite model
            optimization_result = self._edge_optimizer.optimize_whisper(
                model_size=self.model_size,
                language=self.language,
                mobile_export=MobileFramework.TFLITE
            )
            
            if "error" in optimization_result:
                logger.error(f"Error optimizing model for TFLite: {optimization_result['error']}")
                logger.warning("Falling back to transformers model")
                return await self._initialize_transformers_model()
            
            tflite_model_path = optimization_result["tflite_model_path"]
        
        logger.info(f"Loading TFLite model from {tflite_model_path}")
        
        # Load the TFLite model
        self._interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self._interpreter.allocate_tensors()
        
        # Get signature runner (for models with signatures)
        try:
            self._runner = self._interpreter.get_signature_runner()
            self._has_signature = True
            logger.info("Using TFLite model with signature")
        except ValueError:
            self._has_signature = False
            logger.info("Using TFLite model without signature")
            
            # Get input/output details for models without signatures
            self._input_details = self._interpreter.get_input_details()
            self._output_details = self._interpreter.get_output_details()
        
        # We need the processor for pre/post-processing
        from transformers import WhisperProcessor
        
        # Check if there's a processor saved with the TFLite model
        processor_path = os.path.join(os.path.dirname(tflite_model_path), "processor")
        if os.path.exists(processor_path):
            logger.info(f"Loading processor from {processor_path}")
            loop = asyncio.get_event_loop()
            self._processor = await loop.run_in_executor(
                None, 
                lambda: WhisperProcessor.from_pretrained(processor_path)
            )
        else:
            # Fall back to downloading the processor
            logger.info("Downloading processor from HuggingFace")
            lang_suffix = f".{self.language}" if self.language else ""
            model_id = f"openai/whisper-{self.model_size}{lang_suffix}"
            
            loop = asyncio.get_event_loop()
            self._processor = await loop.run_in_executor(
                None, 
                lambda: WhisperProcessor.from_pretrained(model_id)
            )
        
        logger.info("TFLite model initialized successfully")
    
    async def _initialize_coreml_model(self):
        """Initialize an Apple Core ML model for transcription."""
        import platform
        if platform.system() != "Darwin":
            logger.warning("Core ML is only supported on macOS/iOS, falling back to transformers")
            return await self._initialize_transformers_model()
        
        logger.info(f"Initializing Core ML model")
        
        # Determine the model path
        if self.model_path:
            # Use provided model path
            coreml_dir = self.model_path
        else:
            # Use EdgeOptimizer to create or get the model
            if self._edge_optimizer is None:
                # Convert string optimization level to enum
                if self.optimization_level:
                    from src.core.utils.edge_optimization import OptimizationLevel, DeviceTarget, MobileFramework
                    opt_level = getattr(OptimizationLevel, self.optimization_level.upper(), 
                                       OptimizationLevel.MEDIUM)
                else:
                    from src.core.utils.edge_optimization import OptimizationLevel, DeviceTarget, MobileFramework
                    opt_level = OptimizationLevel.MEDIUM
                    
                # Convert string target device to enum
                target_dev = getattr(DeviceTarget, self.target_device.upper(), 
                                    DeviceTarget.CPU)
                    
                self._edge_optimizer = EdgeOptimizer(
                    optimization_level=opt_level,
                    target_device=target_dev,
                    cache_dir=os.path.join(os.getcwd(), "models", "optimized")
                )
            
            # Create or get Core ML model
            optimization_result = self._edge_optimizer.optimize_whisper(
                model_size=self.model_size,
                language=self.language,
                mobile_export=MobileFramework.COREML
            )
            
            if "error" in optimization_result:
                logger.error(f"Error optimizing model for Core ML: {optimization_result['error']}")
                logger.warning("Falling back to transformers model")
                return await self._initialize_transformers_model()
            
            coreml_dir = optimization_result["coreml_model_path"]
        
        logger.info(f"Loading Core ML models from {coreml_dir}")
        
        # Try to import Core ML tools for validation
        try:
            import coremltools as ct
        except ImportError:
            logger.warning("coremltools not available, but proceeding with Core ML Runtime")
        
        # We need to use the Core ML runtime - in a real macOS/iOS app this would use the Core ML framework
        # Here we'll use a simplified version for demonstration purposes
        import json
        config_path = os.path.join(coreml_dir, "whisper_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._coreml_config = json.load(f)
        else:
            self._coreml_config = {
                "encoder_path": "WhisperEncoder.mlpackage",
                "decoder_path": "WhisperDecoder.mlpackage",
                "max_length": 448
            }
        
        # Load tokenizer
        from transformers import WhisperProcessor
        
        # Check for a saved tokenizer
        tokenizer_path = os.path.join(coreml_dir, "tokenizer")
        if os.path.exists(tokenizer_path):
            logger.info(f"Loading processor from {tokenizer_path}")
            loop = asyncio.get_event_loop()
            self._processor = await loop.run_in_executor(
                None, 
                lambda: WhisperProcessor.from_pretrained(tokenizer_path)
            )
        else:
            # Fall back to downloading the processor
            logger.info("Downloading processor from HuggingFace")
            lang_suffix = f".{self.language}" if self.language else ""
            model_id = f"openai/whisper-{self.model_size}{lang_suffix}"
            
            loop = asyncio.get_event_loop()
            self._processor = await loop.run_in_executor(
                None, 
                lambda: WhisperProcessor.from_pretrained(model_id)
            )
        
        # Store model paths for later use
        self._encoder_model_path = os.path.join(coreml_dir, self._coreml_config["encoder_path"])
        self._decoder_model_path = os.path.join(coreml_dir, self._coreml_config["decoder_path"])
        
        # Note: In a real app, you would load these with Core ML:
        # self._encoder_model = MLModel(contentsOf: URL(fileURLWithPath: self._encoder_model_path))
        # self._decoder_model = MLModel(contentsOf: URL(fileURLWithPath: self._decoder_model_path))
        
        # For now, we'll set a flag indicating the models exist but won't load them
        # since we can't actually run them in Python (would need Swift/Objective-C)
        self._coreml_models_available = (
            os.path.exists(self._encoder_model_path) and 
            os.path.exists(self._decoder_model_path)
        )
        
        if self._coreml_models_available:
            logger.info("Core ML models detected and available")
        else:
            logger.warning("Core ML model files not found, will fall back to transformers")
        
        logger.info("Core ML setup completed")
    
    async def _transcribe_with_tflite(self, inputs: Any) -> str:
        """
        Transcribe audio using TensorFlow Lite model.
        
        Args:
            inputs: Preprocessed audio inputs
            
        Returns:
            Transcription text
        """
        import numpy as np
        
        logger.info("Transcribing with TensorFlow Lite model")
        
        # Process inputs
        if isinstance(inputs, str):
            # If input is an audio file path, load and preprocess it
            preprocessed = await self._preprocess_audio(inputs)
            if hasattr(preprocessed, "input_features"):
                input_features = preprocessed.input_features.numpy()
            else:
                input_features = preprocessed
        else:
            # Assume inputs are already preprocessed features
            if hasattr(inputs, "input_features"):
                input_features = inputs.input_features.numpy()
            else:
                input_features = inputs
        
        # Run inference
        if self._has_signature:
            # Use signature runner API
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: self._runner(input_features=input_features)
            )
            
            # Extract output (usually "sequences")
            output_key = "sequences" if "sequences" in outputs else list(outputs.keys())[0]
            generated_ids = outputs[output_key]
        else:
            # Use raw TFLite API
            input_details = self._input_details
            output_details = self._output_details
            
            def run_inference():
                self._interpreter.set_tensor(input_details[0]["index"], input_features)
                self._interpreter.invoke()
                return self._interpreter.get_tensor(output_details[0]["index"])
            
            loop = asyncio.get_event_loop()
            generated_ids = await loop.run_in_executor(None, run_inference)
        
        # Decode the output
        try:
            transcription = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            logger.error(f"Error decoding output: {e}")
            # If decoding fails, return a simplified interpretation
            transcription = f"[TFLite output not decodable - raw shape: {generated_ids.shape}]"
        
        return transcription
    
    async def _transcribe_with_coreml(self, inputs: Any) -> str:
        """
        Transcribe audio using Core ML model.
        
        Note: This is only a demonstration implementation. In a real application,
        this would be implemented in Swift/Objective-C using the Core ML framework.
        
        Args:
            inputs: Preprocessed audio inputs
            
        Returns:
            Transcription text
        """
        logger.info("Transcribing with Core ML model")
        
        if not self._coreml_models_available:
            logger.warning("Core ML models not available, falling back to transformers")
            return await self._transcribe_with_transformers(inputs)
        
        # In a Python environment, we can't actually run Core ML models
        # since they require the macOS/iOS Core ML runtime
        # We'll return a message indicating this
        
        return "[Core ML transcription would run here - requires macOS/iOS app implementation]"
    
    async def _process_message(self, message: Message) -> Optional[Message]:
        """
        Process an incoming message for audio transcription.
        
        Args:
            message: The message to process, should contain audio data
            
        Returns:
            A new message containing the transcription result
        """
        try:
            # Extract audio file path
            message_data = message.data
            if not message_data or "audio_file" not in message_data:
                return Message(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    data={"error": "No audio file provided"},
                    message_type="error"
                )
                
            audio_file = message_data["audio_file"]
            if not os.path.exists(audio_file):
                return Message(
                    sender_id=self.agent_id,
                    recipient_id=message.sender_id,
                    data={"error": f"Audio file not found: {audio_file}"},
                    message_type="error"
                )
                
            # Initialize model if needed
            await self._initialize_model()
            
            # Preprocess audio
            start_time = time.time()
            inputs = await self._preprocess_audio(audio_file)
            preprocess_time = time.time() - start_time
            
            # Transcribe based on model type
            start_time = time.time()
            transcription = await self._transcribe_with_optimized(inputs)
            transcription_time = time.time() - start_time
            total_time = preprocess_time + transcription_time
            
            # Create response message
            result = {
                "transcription": transcription,
                "audio_file": audio_file,
                "model_size": self.model_size,
                "model_type": self.model_type,
                "language": self.language if self.language else "auto",
                "metrics": {
                    "preprocess_time_s": preprocess_time,
                    "transcription_time_s": transcription_time,
                    "total_time_s": total_time
                }
            }
            
            # Unload model if needed
            await self._unload_model_if_inactive()
            
            return Message(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                data=result,
                message_type="transcription_result"
            )
            
        except Exception as e:
            logger.error(f"Error processing transcription request: {e}")
            return Message(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                data={"error": str(e)},
                message_type="error"
            )
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get the current status of the agent."""
        status = await super().get_agent_status()
        
        # Add transcription-specific status
        mobile_info = None
        if self.model_type == ModelType.WHISPER_TFLITE:
            mobile_info = {
                "framework": "TensorFlow Lite",
                "model_path": getattr(self, "model_path", "auto-generated"),
                "has_signature": getattr(self, "_has_signature", False)
            }
        elif self.model_type == ModelType.WHISPER_COREML:
            mobile_info = {
                "framework": "Core ML",
                "model_path": getattr(self, "model_path", "auto-generated"),
                "models_available": getattr(self, "_coreml_models_available", False)
            }
        
        status.update({
            "model_size": self.model_size,
            "model_type": self.model_type,
            "language": self.language if self.language else "auto",
            "model_loaded": self._model is not None,
            "model_cache_enabled": self.use_model_caching,
            "optimization_info": {
                "edge_optimization_available": EDGE_OPTIMIZATION_AVAILABLE,
                "optimization_level": self.optimization_level,
                "target_device": self.target_device
            } if self.model_type in [ModelType.WHISPER_OPTIMIZED, ModelType.WHISPER_OPTIMIZED_ALL_IN_ONE] else None,
            "mobile_info": mobile_info
        })
        
        return status 