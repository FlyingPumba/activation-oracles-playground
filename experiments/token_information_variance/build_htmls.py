"""
Build HTML visualizations of token-level harmfulness scores.

Creates static HTML pages that color each token based on its P(Harmful) score.
"""

import pickle
from pathlib import Path


RESULTS_DIR = Path("/root/activation-oracles/experiments/token_information_variance")
DATA_PATH = RESULTS_DIR / "token_data.pkl"


def get_color(prob: float) -> str:
    """
    Get RGB color for a probability value.
    0.0 (benign) -> Blue
    0.5 (neutral) -> Yellow
    1.0 (harmful) -> Red
    """
    if prob < 0.5:
        # Blue to Yellow (0 -> 0.5)
        t = prob * 2  # 0 to 1
        r = int(t * 255)
        g = int(t * 255)
        b = int((1 - t) * 255)
    else:
        # Yellow to Red (0.5 -> 1)
        t = (prob - 0.5) * 2  # 0 to 1
        r = 255
        g = int((1 - t) * 255)
        b = 0

    return f"rgb({r},{g},{b})"


def build_html(prompt_type: str, data: dict) -> str:
    """Build HTML for one prompt."""
    prompt = data["prompt"]
    response_text = data["response_text"]
    token_data = data["token_data"]

    # Build token spans
    token_html = []
    for td in token_data:
        token_text = td["token_text"]
        harmful_prob = td["harmful_prob"]
        color = get_color(harmful_prob)

        # Escape HTML special characters
        token_text_escaped = (
            token_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
            .replace("\n", "↵\n")  # Show newlines
        )

        token_html.append(
            f'<span class="token" style="background-color: {color};" '
            f'title="P(Harmful): {harmful_prob:.4f}">'
            f'{token_text_escaped}</span>'
        )

    tokens_joined = "".join(token_html)

    # Statistics
    harmful_probs = [td["harmful_prob"] for td in token_data]
    mean_prob = sum(harmful_probs) / len(harmful_probs)
    min_prob = min(harmful_probs)
    max_prob = max(harmful_probs)
    n_harmful = sum(1 for p in harmful_probs if p > 0.5)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Token Harmfulness: {prompt_type.title()}</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            max-width: 1200px;
            margin: 40px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .header {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        h1 {{
            color: #333;
            margin: 0 0 10px 0;
        }}

        .prompt-box {{
            background-color: #e8f4f8;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #2196F3;
            margin: 20px 0;
        }}

        .prompt-label {{
            font-weight: bold;
            color: #666;
            margin-bottom: 5px;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .stat-box {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}

        .stat-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}

        .legend {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .gradient-bar {{
            height: 40px;
            background: linear-gradient(to right,
                rgb(0,0,255),
                rgb(255,255,0) 50%,
                rgb(255,0,0));
            border-radius: 5px;
            margin: 10px 0;
        }}

        .gradient-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: #666;
        }}

        .content {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            line-height: 1.8;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .token {{
            padding: 2px 0;
            cursor: help;
            transition: transform 0.1s;
        }}

        .token:hover {{
            transform: scale(1.1);
            box-shadow: 0 0 5px rgba(0,0,0,0.3);
            z-index: 10;
            position: relative;
        }}

        .info {{
            background-color: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Token-Level Harmfulness Assessment: {prompt_type.title()} Prompt</h1>
        <p style="color: #666; margin: 5px 0 0 0;">
            Activation Oracle assessment of harmfulness for each token in the response
        </p>
    </div>

    <div class="prompt-box">
        <div class="prompt-label">User Prompt:</div>
        <div style="font-weight: bold; color: #333;">{prompt}</div>
    </div>

    <div class="stats">
        <div class="stat-box">
            <div class="stat-label">Tokens</div>
            <div class="stat-value">{len(token_data)}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Mean P(Harmful)</div>
            <div class="stat-value">{mean_prob:.3f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Range</div>
            <div class="stat-value">{min_prob:.3f} - {max_prob:.3f}</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Harmful Tokens (>0.5)</div>
            <div class="stat-value">{n_harmful} ({n_harmful/len(harmful_probs)*100:.1f}%)</div>
        </div>
    </div>

    <div class="legend">
        <h3 style="margin-top: 0;">Color Scale</h3>
        <div class="gradient-bar"></div>
        <div class="gradient-labels">
            <span><strong>0.0</strong> (Benign)</span>
            <span><strong>0.5</strong> (Neutral)</span>
            <span><strong>1.0</strong> (Harmful)</span>
        </div>
    </div>

    <div class="info">
        <strong>💡 Tip:</strong> Hover over any token to see its exact P(Harmful) score
    </div>

    <div class="content">{tokens_joined}</div>

    <div style="margin-top: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 5px; font-size: 12px; color: #666;">
        <strong>Experiment Details:</strong><br>
        Model: Gemma-3-27B-it (Layer 31, 50% depth)<br>
        AO Query: "Is the user's request harmful? Answer only Yes or No."<br>
        Each token's activation was queried independently to extract P(Harmful)
    </div>
</body>
</html>
"""

    return html


def main():
    print("Loading data...")
    with open(DATA_PATH, "rb") as f:
        results = pickle.load(f)

    print("Building HTML pages...")

    # Build HTML for harmful prompt
    harmful_html = build_html("harmful", results["data"]["harmful"])
    harmful_path = RESULTS_DIR / "harmful_tokens.html"
    harmful_path.write_text(harmful_html, encoding="utf-8")
    print(f"✓ Created {harmful_path}")

    # Build HTML for benign prompt
    benign_html = build_html("benign", results["data"]["benign"])
    benign_path = RESULTS_DIR / "benign_tokens.html"
    benign_path.write_text(benign_html, encoding="utf-8")
    print(f"✓ Created {benign_path}")

    print("\nDone! Open the HTML files in a browser to view the visualizations.")


if __name__ == "__main__":
    main()
