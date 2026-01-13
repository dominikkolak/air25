import gradio as gr
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.a2.pipeline import run_pipeline
from src.a1.pipeline import run_pipeline as run_pipeline_a1

def verify_claim_interface(claim, method, k_retrieve, k_vote, use_weighted):
    claims_df = pd.DataFrame({
        "claim": [claim],
        "claim_id": [1]
    })

    if method == "A2":
        results = run_pipeline(
            claims_df=claims_df,
            load_embeddings=True,
            k_retrieve=k_retrieve,
            k_vote=k_vote,
            use_weighted=use_weighted
        )
    else:
        results = run_pipeline_a1(
            claims_df=claims_df,
            k_retrieve=k_retrieve,
            k_vote=k_vote
        )

    result = results[0]

    verdict = result['verdict']
    votes = result['votes']

    votes_text = f"SUPPORTS: {votes.get('SUPPORTS', 0)} | REFUTES: {votes.get('REFUTES', 0)} | NOT ENOUGH INFO: {votes.get('NOT ENOUGH INFO', 0)}"

    evidence_text = ""
    for i, evidence_item in enumerate(result['evidence'], 1):
        if isinstance(evidence_item, tuple):
            evidence, score = evidence_item
            evidence_text += f"[{i}] {evidence}\n\n"
        else:
            evidence_text += f"[{i}] {evidence_item}\n\n"

    return verdict, votes_text, evidence_text


with gr.Blocks(title="Claim Verification") as demo:
    gr.Markdown("Advanced Information Retrieval: Claim Verification")

    claim_input = gr.Textbox(label="Claim", lines=2)

    with gr.Row():
        method = gr.Radio(["A1", "A2"], label="Method", value="A1")
        use_weighted = gr.Checkbox(label="Weighted Voting", value=True)

    with gr.Row():
        k_retrieve = gr.Slider(5, 50, value=20, step=1, label="Retrieve")
        k_vote = gr.Slider(3, 20, value=10, step=1, label="Vote")

    verify_btn = gr.Button("Verify")
    verdict_output = gr.Textbox(label="Verdict", interactive=False)
    votes_output = gr.Textbox(label="Votes", interactive=False)
    evidence_output = gr.Textbox(label="Evidence", lines=15, interactive=False)

    verify_btn.click(
        fn=verify_claim_interface,
        inputs=[claim_input, method, k_retrieve, k_vote, use_weighted],
        outputs=[verdict_output, votes_output, evidence_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)