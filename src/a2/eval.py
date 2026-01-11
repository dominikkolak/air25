import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score

def evaluate_voting_results(results, claims_path):
    mappings_df = pd.read_csv(claims_path)

    gt_lookup = dict(zip(mappings_df['claim_id'], mappings_df['claim_label'].str.upper().str.strip()))

    y_true = []
    y_pred = []

    for result in results:
        claim_id = result['claim_id']
        pred_label = str(result["verdict"]).upper().strip()

        if claim_id in gt_lookup:
            y_true.append(gt_lookup[claim_id])
            y_pred.append(pred_label)
        else:
            print(f"Claim id {claim_id} not found in claims file.")

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred)

    return acc, f1, report, y_pred, y_true
