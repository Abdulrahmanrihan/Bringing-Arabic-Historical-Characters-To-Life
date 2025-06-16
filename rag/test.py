import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

from RAG_2 import ArabicWikiRAG

def evaluate_rag_system(rag_system, character_name, eval_questions, gold_answers, gold_sources):
    latencies = []
    retrieval_precisions = []
    retrieval_recalls = []
    f1s = []
    gen_scores = []

    for i, (question, gold_answer, gold_srcs) in enumerate(zip(eval_questions, gold_answers, gold_sources)):
        start_time = time.time()
        result = rag_system.ask_question(character_name, question)
        latency = time.time() - start_time
        latencies.append(latency)

        # Retrieval evaluation
        retrieved_sources = set(result["sources"])
        gold_sources_set = set(gold_srcs)
        true_positives = len(retrieved_sources & gold_sources_set)
        precision = true_positives / len(retrieved_sources) if retrieved_sources else 0
        recall = true_positives / len(gold_sources_set) if gold_sources_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        retrieval_precisions.append(precision)
        retrieval_recalls.append(recall)
        f1s.append(f1)

        # Generation evaluation: simple ROUGE-L or semantic similarity (here, cosine similarity)
        # For demo, use cosine similarity of embeddings (requires rag_system.embeddings)
        answer_emb = rag_system.embeddings.embed_query(result["answer"])
        gold_emb = rag_system.embeddings.embed_query(gold_answer)
        sim = cosine_similarity([answer_emb], [gold_emb])[0][0]
        gen_scores.append(sim)

        print(f"Q{i+1}: Latency={latency:.2f}s, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, GenSim={sim:.2f}")

    # Plotting
    x = np.arange(1, len(eval_questions)+1)
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(x, latencies, marker='o', label='Latency (s)')
    plt.title('Latency per Question')
    plt.xlabel('Question #')
    plt.ylabel('Seconds')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(x, retrieval_precisions, marker='o', label='Precision')
    plt.plot(x, retrieval_recalls, marker='s', label='Recall')
    plt.plot(x, f1s, marker='^', label='F1 Score')
    plt.plot(x, gen_scores, marker='x', label='Gen Similarity')
    plt.title('Retrieval & Generation Metrics')
    plt.xlabel('Question #')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nAverage Latency: {:.2f}s".format(np.mean(latencies)))
    print("Average Precision: {:.2f}".format(np.mean(retrieval_precisions)))
    print("Average Recall: {:.2f}".format(np.mean(retrieval_recalls)))
    print("Average F1: {:.2f}".format(np.mean(f1s)))
    print("Average Generation Similarity: {:.2f}".format(np.mean(gen_scores)))

# Example usage
if __name__ == "__main__":
    GEMINI_API_KEY = "YOUR_API_KEY"
    rag_system = ArabicWikiRAG(GEMINI_API_KEY, user_id="eval_user")
    character_name = "صلاح الدين الأيوبي"

    eval_questions = [
        "ما هي أهم إنجازاتك؟",
        "كيف حققت هذه الإنجازات؟",
        "ما هي أهم المعارك التي خضتها؟"
    ]
    gold_answers = [
        "من أهم إنجازاتي تحرير القدس...",
        "حققت هذه الإنجازات من خلال...",
        "أهم المعارك كانت معركة حطين..."
    ]
    gold_sources = [
        ["Wikipedia: صلاح الدين الأيوبي"],
        ["Wikipedia: صلاح الدين الأيوبي"],
        ["Wikipedia: صلاح الدين الأيوبي"]
    ]

    evaluate_rag_system(rag_system, character_name, eval_questions, gold_answers, gold_sources)
