import torch

def analyze_sentiment(texts, model, tokenizer, label_encoder, device):
    """Analyze sentiment of input texts using the trained model.
    
    Args:
        texts (list): List of text strings to analyze
        model: Trained model
        tokenizer: Tokenizer for the model
        label_encoder: Label encoder for converting predictions to labels
        device: Device to run the model on (CPU or GPU)
        
    Returns:
        tuple: (predicted_labels, confidences) - arrays of predicted labels and confidence scores
    """
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs[0], dim=1).cpu().numpy()
        confidences = torch.softmax(outputs[0], dim=1).cpu().numpy()
        
    predicted_labels = label_encoder.inverse_transform(predictions)
    max_confidences = [confidences[i][predictions[i]] for i in range(len(predictions))]
    
    return predicted_labels, max_confidences