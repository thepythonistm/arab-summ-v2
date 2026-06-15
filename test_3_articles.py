#!/usr/bin/env python3
"""
Test pipeline on 3 sample articles and print results.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utils FIRST to trigger monkey-patch before any transformers code
import src.utils

from src.hybrid import HybridPipeline

# Auto-discover model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = src.utils.find_model_dir(os.path.join(BASE_DIR, "models", "arat5-xlsum-best"))

SAMPLE_ARTICLES = [
    "أعلنت السلطات اليوم عن إطلاق مشروع جديد للطاقة الشمسية في الصحراء الغربية. يهدف المشروع إلى توفير الكهرباء لأكثر من مليون منزل. وقال وزير الطاقة إن هذا المشروع سيكون الأكبر من نوعه في المنطقة.",
    "فاز المنتخب الوطني لكرة القدم على نظيره المصري بهدفين مقابل هدف واحد. أحرز الهدف الأول اللاعب أحمد في الدقيقة 25، ثم أضاف اللاعب محمد الهدف الثاني في الدقيقة 78. بهذا الفوز تأهل المنتخب إلى الدور نصف النهائي.",
    "أطلقت شركة التكنولوجيا العملاقة هاتفها الذكي الجديد الذي يتميز ببطارية تدوم 48 ساعة. يأتي الهاتف بشاشة مقاس 6.7 بوصة ومعالج من الجيل الخامس. وسيبدأ سعر الهاتف من 899 دولاراً.",
]


def main():
    pipe = HybridPipeline(MODEL_PATH)

    for i, article in enumerate(SAMPLE_ARTICLES, 1):
        print(f"\n{'='*60}")
        print(f"ARTICLE {i}: {article[:120]}...")
        result = pipe.run(article)
        print(f"\nEXTRACTIVE: {result['extractive_output'][:100]}...")
        print(f"\nABSTRACTIVE: {result['abstractive_summary']}")
        print(f"\nFINAL: {result['final_summary']}")
        print(f"Compression: {result['compression_ratio']}%")


if __name__ == "__main__":
    main()