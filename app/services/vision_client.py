# app/services/vision_client.py
import json
from pydantic import BaseModel
from openai import OpenAI
from ..core.config import settings

class FSHDResult(BaseModel):
    risk_probability: float  # 0.0 ~ 1.0
    advice: str

class VisionModelClient:
    def __init__(self):
        api_key = settings.OPENAI_API_KEY or settings.SILICONFLOW_API_KEY
        base_url = settings.OPENAI_BASE_URL or settings.SILICONFLOW_BASE_URL
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def infer_from_base64(self, base64_str: str) -> FSHDResult:
        prompt = """
        你是一位专业的神经内科医生，擅长 FSHD（面肩肱型肌营养不良）影像诊断。
        请分析上传的背部/肩胛照片，判断是否存在典型 FSHD 特征（如翼状肩胛、面部不对称等）。

        **重要：请严格只返回以下格式的 JSON，不要添加任何其他文字、解释、Markdown 或空格！**
        {
          "risk_probability": 0.0 到 1.0 的浮点数，表示患病风险概率
          "advice": "一句话诊疗建议，如'风险较高，建议转诊神经内科'或'未见明显特征，可继续观察'"
        }
        """

        response = self.client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            # 可备选: "Qwen/Qwen-VL-Plus", "OpenGVLab/InternVL2-26B"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=512,
        )

        content = response.choices[0].message.content.strip()
        try:
            data = json.loads(content)
            return FSHDResult(**data)
        except Exception as e:
            print(f"JSON 解析失败: {e} - 原始内容: {content}")
            return FSHDResult(risk_probability=0.0, advice="模型解析失败，请重试")
