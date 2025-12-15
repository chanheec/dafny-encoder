import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import os, json

# ==========================================
# Configuration
# ==========================================
DATA_PATH = "/content/drive/MyDrive/dafny-encoder/dafny-data.json"
MODEL_PATH = "/content/drive/MyDrive/dafny-encoder/dafny_lemma_recommender_bce"
# MODEL_PATH = "/content/drive/MyDrive/dafny-encoder/dafny_lemma_recommender"
OUTPUT_PATH = "/content/drive/MyDrive/dafny-encoder/recommendation_results_v2.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ==========================================
# Eval data Preparation
# ==========================================
def load_dafny_data(json_file_path):
    """
    returns
    (1) list of (goal-req/ens, function-body) pairs
    (2) the long list of candidate lemmas 
    (3) from (2), simply include the name of the lemma for easier matching later
    """
    print(json_file_path)
    examples = []
    if not os.path.exists(json_file_path):
        print(f"Data file {json_file_path} not found. Skipping.")
        return []

    # TODO: name, pre, body, post ----  different pair structures might work better
    with open(json_file_path, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
        lemma_pool = []
        lemma_names = []
        # Example: A proof needing a math lemma
        # Example: A list proof
        for pair in pairs:
            try:
                method_name = pair["method_name"]
                body = pair["body"]
                body = body.strip()

                name_body = f"Method: {method_name}\nBody: {body}"
                spec = f"Method: {method_name}"  # function name and type signature is good to include in search

                for pre in pair["preconditions"]:
                    pre = pre.strip()
                    spec = f"{spec}\nrequires {pre}"

                for postcond in pair["postconditions"]:
                    postcond = postcond.strip()
                    spec = f"{spec}\nensures {pre}"
                
                examples.append((spec, body))
                lemma_pool.append(name_body)
                lemma_names.append(method_name)

            except json.JSONDecodeError:
                print("json error")
                continue

    print(examples[0])
    return examples, lemma_pool, lemma_names


def load_eval_data(json_file_path):
    """
    filters the output of `load_dafny_data` for eval purposes
    """

    goals_and_body, candidate_lemmas, lemma_names = load_dafny_data(json_file_path)
    n = len(candidate_lemmas)
    # use last 1000 lemmas as candidates as lemma pool
    if n > 1000:
        candidate_lemmas = candidate_lemmas[-1000:]
        lemma_names = lemma_names[-1000:]
    else: 
        # die
        print("Not enough lemmas found. Please ensure the dataset contains sufficient examples.")
        exit(1)

    # use last 230 goals as eval goals (AWS crypto repo is at the last 230 dataset examples)
    if len(goals_and_body) > 230:
        goals_and_body = goals_and_body[-230:]
    else: 
        # die
        print("Not enough eval goals found. Please ensure the dataset contains sufficient examples.")
        exit(1)

    return goals_and_body, candidate_lemmas, lemma_names



# ==========================================
# Model Loading and Scoring
# ==========================================
def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        return tokenizer, model
    except OSError:
        print(f"Error: Model not found at {MODEL_PATH}. Please run the training script first.")
        exit(1)


def get_relevance_score(model, tokenizer, goal, lemma):
    """
    Returns a float score indicating how relevant the lemma is to the proof goal.
    Higher score ---> more relevant.
    """
    # Format: <s> Context </s> </s> Lemma </s>
    inputs = tokenizer(
        goal,
        lemma,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():  # no gradient computation 
        outputs = model(**inputs)
        logits = outputs.logits
        if model.config.problem_type == "regression":
            score = logits.item()
        else:
            prob = torch.sigmoid(logits).item()
            score = prob

        # MSE Loss: single regression output
        # score = outputs.logits.item()  # single output for relevance score (regression)

        # BCE Loss: with single logit output
        # logit = outputs.logits.item()
        # prob = torch.sigmoid(outputs.logits).item() 
    return score



# ==========================================
# Main Execution
# ==========================================
def main():
    tokenizer, model = load_model()
    goals_and_body, candidate_lemmas, lemma_names = load_eval_data(DATA_PATH)
    goals = [goal for goal, body in goals_and_body]
    bodies = [body for goal, body in goals_and_body]

    total_goals = len(goals)
    used_lemma_recovery_count = 0 # success count: number of goals where a used lemma is found in top recommendations
    original_body_recovery_count = 0 # number of goals where the original body is found in top recommendations (this score is expected to be very high, as the original body is obviously relevant to its own goal)
    no_useful_lemma_in_pool_count = 0 # failure count: no useful lemma is in the candidate pool at all

    print("\n" + "="*50)
    print("Begin Dafny Lemma Recommendation Evaluation")
    print("="*50)

    for i, goal in enumerate(goals):
        print(f"\nScanning Lemma for Goal {i+1}:")
        print("-" * 20)
        print(goal.strip().split('\n')[0] + " ...") # Print first line of goal for brevity
        print("-" * 20)

        # 1. Score all candidates against this context
        # TODO: this is quite slow; would be great to batch if possible
        scored_candidates = []
        for idx, lemma in enumerate(candidate_lemmas):
            score = get_relevance_score(model, tokenizer, goal, lemma)
            scored_candidates.append((lemma, score, lemma_names[idx]))

        # 2. Sort by score (Highest first)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # 3. Print Results and save to output file
        print("Recommended Lemmas (Ranked by Relevance):")
        out_file = open(OUTPUT_PATH, 'a', encoding='utf-8')
        out_file.write(f"\nGoal {i+1}:\n")
        out_file.write(goal + "\n")
        out_file.write("Recommended Lemmas (Ranked by Relevance):\n")
        for rank, (lemma, score, lemma_name) in enumerate(scored_candidates):
            # Simple visual indicator
            bar = "█" * int(score * 10) if score > 0 else "" 
            line = f"{rank+1}. [{score:.4f}] {lemma[:60]}... {bar}\n"
            out_file.write(line)
            print(line.strip())
            if rank >= 9: 
                break  # Limit to top 10 recommendations

        # 4. Analyze if the original body is in top recommendations
        body = bodies[i] # original proof body corresponding to the goal
        original_body_in_recommendations = any(body in lemma for lemma, score, lemma_name in scored_candidates[:10])
        if original_body_in_recommendations:
            original_body_recovery_count += 1
        else:
            print("-> Original body NOT found in top 10 recommendations.")
            out_file.write("-> Original body NOT found in top 10 recommendations.\n")

        # 5. Check if there is used lemma in top 10 recommendations
        # TODO: this is a simple string match. While this would still be quite accurate, it can be improved with actual parsing
        recommended_used_lemma = False
        for rank, (lemma, score, lemma_name) in enumerate(scored_candidates):
            if '(' in lemma_name:
                lemma_name = lemma_name.split("(")[0]
            lemma_call_str = f" {lemma_name}("
            if lemma_call_str in body:
                recommended_used_lemma = True
                print(f"-> Rank '{rank}': Useful lemma '{lemma_name}' from the pool is used in this goal.")
                out_file.write(f"-> Rank '{rank}': Useful lemma '{lemma_name}' from the pool is used in this goal.\n")
            if rank >= 9:
                break  # Limit to top 10 recommendations
        if recommended_used_lemma:
            used_lemma_recovery_count += 1    
        
        # 6. check if *any* candidate lemma is used in the goal body (i.e., if recommendation is even possible)
        used_lemma_in_goal = False
        for lemma_name in lemma_names:
            if '(' in lemma_name:
                lemma_name = lemma_name.split("(")[0]
            lemma_call_str = f" {lemma_name}("
            if lemma_call_str in body:
                used_lemma_in_goal = True
                break

        if not used_lemma_in_goal:
            no_useful_lemma_in_pool_count += 1
            print("-> No useful lemma from the pool is used in this goal.")
            out_file.write("-> No useful lemma from the pool is used in this goal.\n")
                      

    success_possible_count = total_goals - no_useful_lemma_in_pool_count
    success_rate = (used_lemma_recovery_count / success_possible_count * 100) if success_possible_count > 0 else -1.0

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total Goals Evaluated: {len(goals)}")
    print(f"Used Lemma Recovery in Top 10: {used_lemma_recovery_count} ({(used_lemma_recovery_count/len(goals))*100:.2f}%)")
    print(f"Original Body Recovery in Top 10: {original_body_recovery_count} ({(original_body_recovery_count/len(goals))*100:.2f}%)")
    print(f"Goals with No Useful Lemma in Pool: {no_useful_lemma_in_pool_count} ({(no_useful_lemma_in_pool_count/len(goals))*100:.2f}%)")
    print(f"Result: Success Rate (Used Lemma Recovery) excluding goals with no useful lemma in pool: {success_rate:.2f}%")

    # write summary to output file
    with open(OUTPUT_PATH, 'a', encoding='utf-8') as out_file:
        out_file.write("\n" + "="*50 + "\n")
        out_file.write("EVALUATION SUMMARY\n")
        out_file.write("="*50 + "\n")
        out_file.write(f"Total Goals Evaluated: {len(goals)}\n")
        out_file.write(f"Used Lemma Recovery in Top 10: {used_lemma_recovery_count} ({(used_lemma_recovery_count/len(goals))*100:.2f}%)\n")
        out_file.write(f"Original Body Recovery in Top 10: {original_body_recovery_count} ({(original_body_recovery_count/len(goals))*100:.2f}%)\n")
        out_file.write(f"Goals with No Useful Lemma in Pool: {no_useful_lemma_in_pool_count} ({(no_useful_lemma_in_pool_count/len(goals))*100:.2f}%)\n")
        out_file.write(f"Result: Success Rate (Used Lemma Recovery) excluding goals with no useful lemma in pool: {success_rate:.2f}%\n")


if __name__ == "__main__":
    main()