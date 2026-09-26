import json
import time

import streamlit as st
import torch
from huggingface_hub import scan_cache_dir
from transformers import AutoModelForCausalLM, AutoTokenizer

# Own tag format for base models, as in notebooks/transformers.ipynb
TAG_TEMPLATE = """<INSTRUCTION>
{system}
</INSTRUCTION>

<HUMAN>
{user}
</HUMAN>

<BOT>
"""


def cached_text_generation_models() -> list[str]:
    models = []
    for repo in scan_cache_dir().repos:
        if repo.repo_type != "model":
            continue
        for revision in repo.revisions:
            files = {f.file_name for f in revision.files}
            has_weights = any(name.endswith((".safetensors", ".bin")) for name in files)
            if "config.json" not in files or not has_weights:
                continue
            config = json.loads((revision.snapshot_path / "config.json").read_text())
            if any(arch.endswith("ForCausalLM") for arch in config.get("architectures", [])):
                models.append(repo.repo_id)
                break
    return sorted(models)


@st.cache_resource
def load_model(model_id: str):
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model, time.perf_counter() - start


st.set_page_config(page_title="LLM Playground", page_icon="🧪", layout="wide")
st.title("LLM Playground 🧪")

models = cached_text_generation_models()
if not models:
    st.error("Keine Text-Generation-Modelle im Hugging Face Cache gefunden.")
    st.stop()

with st.sidebar:
    st.header("Modell")
    model_id = st.selectbox("Heruntergeladene Modelle", models)
    tokenizer, model, load_seconds = load_model(model_id)

    st.metric("Parameter", f"{model.num_parameters() / 1e6:,.0f} M")
    st.metric("Speicher (Gewichte)", f"{model.get_memory_footprint() / 2**20:,.0f} MiB")
    st.caption(f"Gerät: `{model.device}`, dtype: `{model.dtype}`, geladen in {load_seconds:.1f} s")

    st.header("Generierung")
    has_chat_template = tokenizer.chat_template is not None
    prompt_format = st.radio(
        "Prompt-Format",
        ["Chat-Template", "Eigene Tags"],
        # the name is only a heuristic: base models may ship a chat template too
        index=0 if has_chat_template and "instruct" in model_id.lower() else 1,
        disabled=not has_chat_template,
    )
    max_new_tokens = st.slider("max_new_tokens", 10, 500, 100)
    do_sample = st.checkbox("Sampling (sonst greedy)")
    temperature = st.slider("temperature", 0.1, 2.0, 0.7, disabled=not do_sample)

system_prompt = st.text_area(
    "System Prompt",
    "You are a helpful and friendly bot.\nYou provide short and concise answers.",
)
user_message = st.text_area("Frage", placeholder="Wer bist du?")

if st.button("Antwort generieren", type="primary"):
    if not user_message:
        st.warning("Bitte zuerst eine Frage eingeben.")
        st.stop()

    if prompt_format == "Chat-Template":
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = TAG_TEMPLATE.format(system=system_prompt, user=user_message)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_args = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        generation_args["temperature"] = temperature
    if prompt_format == "Eigene Tags":
        # stop as soon as the base model closes its answer or starts inventing the next turn
        generation_args |= {"stop_strings": ["</BOT>", "<HUMAN>"], "tokenizer": tokenizer}

    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with st.spinner("Generiere..."):
        output = model.generate(**inputs, **generation_args)
    seconds = time.perf_counter() - start

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    for stop in ("</BOT>", "<HUMAN>"):
        answer = answer.split(stop)[0]

    st.subheader("Antwort")
    st.markdown(answer.strip())

    cols = st.columns(4)
    cols[0].metric("Inferenzzeit", f"{seconds:.2f} s")
    cols[1].metric("Prompt-Tokens", inputs["input_ids"].shape[1])
    cols[2].metric("Neue Tokens", len(new_tokens))
    cols[3].metric("Tokens/s", f"{len(new_tokens) / seconds:.1f}")
    if model.device.type == "cuda":
        st.caption(f"GPU-Speicher Spitze während Inferenz: {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB")

    with st.expander("Prompt, den das Modell gesehen hat"):
        st.code(prompt, language=None)
    with st.expander("Tokens der Antwort"):
        st.write(tokenizer.convert_ids_to_tokens(new_tokens))
