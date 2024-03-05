# Utilities for preprocessing questions
# Code adopted from
# https://github.com/facebookresearch/clevr-iep/blob/master/iep/preprocess.py


SPECIAL_TOKENS = {
    '<NULL>': 0,
    '<START>': 1,
    '<END>': 2,
    '<UNK>': 3,
}

def tokenize(s, delim=' ',
            add_start_token=True, add_end_token=True,
            punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    s= str(s)
    if punct_to_keep is not None:
        for p in punct_to_keep:
            s = s.replace(p, '%s%s' % (delim, p))
            
    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def tokenize_program(s, delim=' ',
            add_start_token=True, add_end_token=True,
            punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    s= str(s)
    #skipping missing(Q):- in the beginning and . in the end
    #print('Tokenizing:', s)
    s = s[12:len(s)-1]
     
    if punct_to_keep is not None:
        for p in punct_to_keep:
            
            s = s.replace(p, '%s%s%s' % (delim, p, delim))
            
    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    tokens_revised = []
    for t in range(len(tokens)):
    	if tokens[t] in ['hasProperty', 'same_material', 'same_size', 'same_color', 'same_shape', 'front', 'behind', 'right', 'left']:
    		tok = tokens[t]
    		t = t+1
    		while(tokens[t] != ')'):
    			tok = tok+tokens[t]
    			t=t+1
    		tok = tok+tokens[t]
    		tokens_revised.append(tok)
    	else:
    		if tokens[t] == '!=':
    			start = t-1
    			while(tokens[start]!=','):
    				start=start-1
    			end = t+1
    			while(end!=len(tokens) and tokens[end]!=',' and tokens[end]!='.'):
    				end=end+1
    			tokens_revised.append(''.join(tokens[start+1:end]).strip())
    if add_start_token:
    	tokens_revised.insert(0, '<START>')
    if add_end_token:
        tokens_revised.append('<END>')
    #print('Tokens:',tokens_revised)
    return tokens_revised


def build_vocab_program(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    token_to_count = {}
    for seq in sequences:
    	s = str(seq)
    	seq_tokens = tokenize_program(seq, delim = ' ', add_start_token=False, add_end_token=False, punct_to_keep = punct_to_keep, punct_to_remove = punct_to_remove)
    	for token in seq_tokens:
            
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)  
    #print("Vocabulory:")
    #print(token_to_idx)
    return token_to_idx



def tokenize_program_char(s, delim=' ',
            add_start_token=True, add_end_token=True,
            punct_to_keep=None, punct_to_remove=None):
    """
    Tokenize a sequence, converting a string s into a list of (string) tokens by
    splitting on the specified delimiter. Optionally keep or remove certain
    punctuation marks and add start and end tokens.
    """
    s= str(s)
    if punct_to_keep is not None:
        for p in punct_to_keep:
            
            s = s.replace(p, '%s%s%s' % (delim, p, delim))
            
    if punct_to_remove is not None:
        for p in punct_to_remove:
            s = s.replace(p, '')

    tokens = s.split(delim)
    
			 	
    if add_start_token:
        tokens.insert(0, '<START>')
    if add_end_token:
        tokens.append('<END>')
    return tokens

def build_vocab_program_char(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    token_to_count = {}
    for seq in sequences:
        seq_tokens = tokenize_program_char(seq, delim = ' ', add_start_token=False, add_end_token=False, punct_to_keep = punct_to_keep, punct_to_remove = punct_to_remove)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)  
    #print("Vocabulory:")
    #print(token_to_idx)
    return token_to_idx

def build_vocab(sequences, min_token_count=1, delim=' ',
                punct_to_keep=None, punct_to_remove=None):
    token_to_count = {}
    for seq in sequences:
        seq_tokens = tokenize(seq, delim = ' ', add_start_token=False, add_end_token=False, punct_to_keep = punct_to_keep, punct_to_remove = punct_to_remove)
        for token in seq_tokens:
            if token not in token_to_count:
                token_to_count[token] = 0
            token_to_count[token] += 1

    token_to_idx = {}
    for token, idx in SPECIAL_TOKENS.items():
        token_to_idx[token] = idx
    for token, count in sorted(token_to_count.items()):
        if count >= min_token_count:
            token_to_idx[token] = len(token_to_idx)  
    #print("Vocabulory:")
    #print(token_to_idx)
    return token_to_idx

def encode(seq_tokens, token_to_idx, allow_unk=False):
    seq_idx = []
    for token in seq_tokens:
        if token not in token_to_idx:
            if allow_unk:
                token = '<UNK>'
            else:
                raise KeyError('Token "%s" not in vocab' % token)
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == '<END>':
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)
