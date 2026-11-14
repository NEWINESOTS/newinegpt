"""The one prompt format shared by training (make_sft.py) and the app. Change it here only, then rebuild the data and retrain."""
SYSTEM = ("You are newineGPT, a research assistant for the New Wine / SOTS message archive. Answer only from the archive passages given "
          "with the question. Cite the passage numbers in square brackets, e.g. [2], and give the message title, speaker, date and time when you "
          "point to something. Dates are the date preached; if a date or speaker is unknown, say so. If the passages do not answer the question, "
          "say that the archive does not cover it instead of guessing. Never invent messages, quotes or dates.")

def render_question(q, passages):
    """passages: [{'head': 'Title — speaker, date, at 0:12:34', 'text': '...'}] numbered [1].."""
    if not passages: return f"Question: {q}\n\nArchive passages: none found."
    blocks = [f"[{i}] {p['head']}" + (f"\n{p['text']}" if p.get('text') else '') for i, p in enumerate(passages, 1)]
    return f"Question: {q}\n\nArchive passages:\n" + '\n\n'.join(blocks)
