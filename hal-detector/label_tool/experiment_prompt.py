CLAIM_LEVEL_GPT_CHECK_PROMPT = {
    "en":"""
        Imagine you are an intelligent teacher. Thoroughly read the instruction:{instruction}, reference answer:{ground_truth} and the prediction answer:{text} to ensure a clear understanding of the information provided. Assess the correctness of the predictions. 
        If the prediction answer does not conflict with the reference answer, please generate “correct”. If the prediction answer conflict with the reference answer, please generate “incorrect”. The output should only be “correct” or “incorrect”. 
        Example:
        Question:
        "The two lines are parallel to each other. Why?"
        Reference answer:
         "The two straight lines in the picture are parallel, because the slopes of two straight lines are equal"
        Prediction answer:
        "The two lines will never intersect."
        Output:
        "correct"
    
        Question:
        {instruction}
        Reference answer:
        {ground_truth}
        Prediction answer:
        {text}
        Output:
        """
}

SENTENCE_LEVEL_GPT_CHECK_PROMPT = {
    "en": """
        Imagine you are an intelligent teacher. Thoroughly read the question, reference answer and the prediction answer to ensure a clear understanding of the information provided. Assess the correctness of the predictions. 
        If the prediction answer does not conflict with the reference answer, please generate **false**. If the prediction answer conflicts with the reference answer, please generate **true**. The output should only be **true** or **false** (without quotes).
        Example:
        Question:
        "How many elephants are in the image?"
        Reference answer:
        "Based on the description provided, it is difficult to determine the exact number of elephants in the image. The sentences mention a group of elephants, a herd of elephants, and very many elephants, but the specific count is not mentioned."
        Prediction answer:
        "There are nine elephants in the image."
        Output:
        "true"

        Question:
        {instruction}
        Reference answer:
        {ground_truth}
        Prediction answer:
        {text}
        Output:
        """
}