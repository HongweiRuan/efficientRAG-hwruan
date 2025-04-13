from transformers import AutoModel, AutoTokenizer

# download Contriever model
AutoModel.from_pretrained("facebook/contriever-msmarco", cache_dir="model_cache/facebook/contriever-msmarco")
AutoTokenizer.from_pretrained("facebook/contriever-msmarco", cache_dir="model_cache/facebook/contriever-msmarco")

# download DeBERTa model (e.g. deberta-base)
AutoModel.from_pretrained("microsoft/deberta-base", cache_dir="model_cache/microsoft/deberta-base")
AutoTokenizer.from_pretrained("microsoft/deberta-base", cache_dir="model_cache/microsoft/deberta-base")