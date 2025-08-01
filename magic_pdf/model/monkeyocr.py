import torch
import yaml
import io
import base64
import asyncio
from loguru import logger
from typing import List, Dict
from openai import OpenAI, AsyncOpenAI
from collections import defaultdict
from transformers import LayoutLMv3ForTokenClassification

from magic_pdf.config.prompts import LoRAType
from magic_pdf.utils.load_image import load_image
from magic_pdf.model.sub_modules.layout_detection.doclayout_yolo import DocLayoutYOLO


class MonkeyOCR:
    def __init__(
        self,
        config_path,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.configs = yaml.load(f, Loader=yaml.FullLoader)
        logger.info("using configs: {}".format(self.configs))

        self.device = self.configs.get("device", "cpu")
        logger.info("using device: {}".format(self.device))

        bf16_supported = False
        if self.device.startswith("cuda"):
            bf16_supported = torch.cuda.is_bf16_supported()
        elif self.device.startswith("mps"):
            bf16_supported = True

        self.models_config = self.configs.get("models")

        ### Layout detection model:
        self.layout_det_config = self.models_config.get("layout_detection")
        self.layout_det = DocLayoutYOLO(
            weight=self.layout_det_config.get("weight"),
            device=self.device,
        )
        logger.info(f'Layout detection model loaded: {self.layout_det_config.get("name")}')
        ### : Layout detection model

        ### Relation model:
        self.relation_config = self.models_config.get("relation")
        _model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.relation_config.get("weight"),
        )
        if bf16_supported:
            _model.to(self.device).eval().bfloat16()
        else:
            _model.to(self.device).eval()
        self.relation = _model
        logger.info(f'Relation model loaded: {self.relation_config.get("name")}')
        ### Relation model:

        ### LLM:
        self.llm_config = self.models_config.get("llm")
        self.llm = GroupedLLM(
            url=self.llm_config.get("url"),
            model_name=self.llm_config.get("name"),
            loras=self.llm_config.get("loras", {}),
        )
        logger.info(f"LLM loaded: {self.llm.model_name}")
        ### : LLM


class LLM:
    def __init__(
        self,
        url: str,
        model_name: str,
        api_key: str = "EMPTY",
        max_tokens: int = 4096,
        temperature: float = 0.
    ):
        self.model_name = model_name
        self.base_url = url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Health check
        try:
            sync_client = OpenAI(base_url=url, api_key=api_key)
            response = sync_client.models.list()
            if not response.data:
                raise ValueError(f"No models found for model name: {self.model_name}")
            # logger.info("API connection validated successfully.")
        except Exception as e:
            logger.error(f"API connection validation failed: {e}")
            raise ValueError(f"Invalid API URL or API key: {e}")

    async def _infer_single(
        self,
        image,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
    ):
        try:
            async_client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )

            # 이미지 로드 및 인코딩
            pil_image = load_image(image, max_size=1600)
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # 메시지 생성
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url", 
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ]
            # API 호출
            model_out = await async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            # print(self.model_name)
            # print(messages[1]["content"][1]["text"])
            # if self.model_name == "table_image_otsl":
            #     print(model_out.choices[0].message.content)
            return model_out.choices[0].message.content

        except Exception as e:
            logger.error(f"Error processing single inference: {e}")
            return f"Error: {str(e)}"

    async def infer_async_batch(
        self,
        images,
        user_prompts,
    ):
        logger.info(f"{self.model_name} - Processing batch inference with {len(images)} images and user prompts.")
        if len(images) != len(user_prompts):
            raise ValueError("Images and user prompts must have the same length")
        
        # 모든 작업을 비동기로 실행
        tasks = [
            self._infer_single(
                image=image,
                user_prompt=user_prompt,
            ) 
            for image, user_prompt in zip(images, user_prompts)
        ]
        # 병렬 실행 (에러가 있어도 다른 것들은 계속)
        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )
        # Exception을 문자열로 변환
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(f"Error processing item {i}: {str(result)}")
            else:
                processed_results.append(result)
        return processed_results


class GroupedLLM:
    def __init__(
        self,
        url: str,
        model_name: str,
        loras: Dict[str, str],
        api_key: str = "EMPTY",
    ):
        self.model_name = model_name

        self.base_model = LLM(
            url=url,
            model_name=model_name,
            api_key=api_key
        )
        self.lora_models = {}
        for lora_type, lora_name in loras.items():
            logger.info(f"Loading LoRA model: {lora_type} -> {lora_name}")
            self.lora_models[lora_type] = LLM(
                url=url,
                model_name=lora_name,
                api_key=api_key
            )

    def get_available_models(
        self,
    ) -> List[str]:
        return [LoRAType.BASE] + list(self.lora_models.keys())

    async def _infer_async_batch(
        self,
        images,
        user_prompts,
        model_types: List = None,
    ) -> List[str]:
        async def process_group(
            model_type,
            group,
        ):
            if model_type == LoRAType.BASE or model_type not in self.lora_models:
                _model = self.base_model
            else:
                _model = self.lora_models[model_type]
            return await _model.infer_async_batch(
                images=group["images"],
                user_prompts=group["user_prompts"],
            )

        if model_types is None:
            model_types = [LoRAType.BASE] * len(images)
        
        if len(model_types) != len(images):
            raise ValueError("model_types length must match images length")

        # 모델별로 그룹핑
        groups = defaultdict(lambda: defaultdict(list))
        for i, (image, user_prompt, model_type) in enumerate(
            zip(
                images,
                user_prompts,
                model_types,
            ),
        ):
            groups[model_type]["images"].append(image)
            groups[model_type]["user_prompts"].append(user_prompt)
            groups[model_type]["indices"].append(i)

        # 모든 그룹을 동시에 처리
        group_tasks = [
            (
                model_type,
                group,
                process_group(
                    model_type=model_type,
                    group=group,
                )
            )
            for model_type, group in groups.items()
        ]
        # 🚀 동시 실행!
        group_results = await asyncio.gather(
            *[task[2] for task in group_tasks],
        )
        # 원래 순서로 결과 재배치
        final_results = [None] * len(images)
        for (model_type, group, _), results in zip(group_tasks, group_results):
            for result, original_idx in zip(results, group["indices"]):
                final_results[original_idx] = result
        return final_results

    def __call__(
        self,
        images,
        user_prompts,
        model_types: List = None,
    ):
        return asyncio.run(
            self._infer_async_batch(
                images,
                user_prompts,
                model_types,
            ),
        )
