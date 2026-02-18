
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import json
import argparse
import os
import re
import httpx
from openai import OpenAI
from bert_score import score as bert_score
from rouge_score import rouge_scorer


def get_openrouter_client(api_key: str) -> OpenAI:

    http_client = httpx.Client(
        timeout=120.0,
        default_encoding="utf-8",
    )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        http_client=http_client,
    )


def call_model(
    client: OpenAI,
    model: str,
    prompt: str,
    reference: str,
    temperature: float = 0.7,
) -> str:

    if "[ВСТАВИТЬ ЮРИДИЧЕСКИЙ ТЕКСТ]" in prompt:
        full_prompt = prompt.replace("[ВСТАВИТЬ ЮРИДИЧЕСКИЙ ТЕКСТ]", reference)
    else:
        full_prompt = f"{prompt}\n\nЭталонный текст:\n{reference}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=temperature,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def compute_bert_score(generated: str, reference: str) -> dict:
    P, R, F1 = bert_score(
        [generated],
        [reference],
        lang="ru",
        verbose=False,
    )
    return {
        "precision": float(P[0]),
        "recall": float(R[0]),
        "f1": float(F1[0]),
    }


def compute_rouge(generated: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
    )
    scores = scorer.score(reference, generated)
    return {
        "rouge1": {"f1": scores["rouge1"].fmeasure},
        "rouge2": {"f1": scores["rouge2"].fmeasure},
        "rougeL": {"f1": scores["rougeL"].fmeasure},
    }


def _count_sentences(text: str) -> int:
    if not text.strip():
        return 0
    parts = re.split(r"[.!?]+", text.strip())
    return max(1, len([p for p in parts if p.strip()]))


def compute_length_metrics(text: str) -> dict:
    words = text.split()
    sentences = _count_sentences(text)
    return {
        "chars": len(text),
        "words": len(words),
        "sentences": sentences,
        "avg_word_length": len(text) / len(words) if words else 0,
        "avg_sentence_length": len(words) / sentences if sentences else 0,
    }


def evaluate(generated: str, reference: str) -> dict:
    bert = compute_bert_score(generated, reference)
    rouge = compute_rouge(generated, reference)
    len_gen = compute_length_metrics(generated)
    len_ref = compute_length_metrics(reference)

    return {
        "bert_score": bert,
        "rouge": rouge,
        "length_generated": len_gen,
        "length_reference": len_ref,
    }


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Оценка перефразирования через OpenRouter с метриками для русского языка"
    )
    parser.add_argument(
        "--config",
        default="config.json",
        help="Путь к файлу конфигурации (JSON)",
    )
    parser.add_argument("--api-key", help="OpenRouter API key (переопределяет config)")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Модели (переопределяет config)",
    )
    parser.add_argument(
        "--prompt",
        help="Промпт (переопределяет prompt_file/prompt из config)",
    )
    parser.add_argument(
        "--reference",
        help="Эталонный текст (переопределяет reference_file/reference из config)",
    )
    parser.add_argument("--output", help="Файл результатов (переопределяет config)")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Ошибка: файл конфигурации {args.config} не найден.")
        return
    except json.JSONDecodeError as e:
        print(f"Ошибка в JSON конфигурации: {e}")
        return

    api_key = args.api_key or cfg.get("api_key")
    models = args.models or cfg.get("models")
    prompt = args.prompt
    if prompt is None:
        prompt_file = cfg.get("prompt_file")
        if prompt_file:
            if not os.path.isabs(prompt_file):
                config_dir = os.path.dirname(os.path.abspath(args.config))
                prompt_file = os.path.join(config_dir, prompt_file)
            with open(prompt_file, encoding="utf-8") as f:
                prompt = f.read().strip()
        else:
            prompt = cfg.get("prompt")
    reference = args.reference
    if reference is None:
        ref_file = cfg.get("reference_file")
        if ref_file:
            if not os.path.isabs(ref_file):
                config_dir = os.path.dirname(os.path.abspath(args.config))
                ref_file = os.path.join(config_dir, ref_file)
            with open(ref_file, encoding="utf-8") as f:
                reference = f.read().strip()
        else:
            reference = cfg.get("reference")
    output = args.output or cfg.get("output", "results.json")
    temperature = cfg.get("temperature", 0.7)

    if not api_key:
        print("Ошибка: api_key не задан (в config или --api-key)")
        return
    if not models:
        print("Ошибка: models не заданы (в config или --models)")
        return
    if not prompt:
        print("Ошибка: prompt не задан (prompt_file, prompt в config или --prompt)")
        return
    if not reference:
        print("Ошибка: reference не задан (reference_file, reference в config или --reference)")
        return

    client = get_openrouter_client(api_key)
    results = []

    for model in models:
        print(f"Запрос к модели {model}...")
        text = None
        try:
            text = call_model(
                client, model, prompt, reference, temperature=temperature
            )
            metrics = evaluate(text, reference)
            results.append(
                {
                    "model": model,
                    "generated_text": text,
                    "metrics": metrics,
                }
            )
            print(f"  Готово. BERT F1: {metrics['bert_score']['f1']:.3f}")
        except Exception as e:
            print(f"  Ошибка: {e}")
            results.append(
                {
                    "model": model,
                    "generated_text": text,
                    "error": str(e),
                    "metrics": None,
                }
            )

    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в {output}")


if __name__ == "__main__":
    main()
