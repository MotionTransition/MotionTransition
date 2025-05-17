import torch

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting


    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}
    if 'style_name' in notnone_batches[0]:
        stylenamebatch = [b['style_name'] for b in notnone_batches]
        cond['y'].update({'style_name': stylenamebatch})

    if 'motion_t' in notnone_batches[0]:
        motiontbatch = [b['motion_t'] for b in notnone_batches]
        cond['y'].update({'motion_t': motiontbatch})

    if 'motion_s_name' in notnone_batches[0]:
        motiontbatch = [b['motion_s_name'] for b in notnone_batches]
        cond['y'].update({'motion_s_name': motiontbatch})

    if 'motion_s' in notnone_batches[0]:
        motiontbatch = [b['motion_s'] for b in notnone_batches]
        cond['y'].update({'motion_s': motiontbatch})

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)

def amass_collate(batch):
    adapted_batch = [{
        'inp': (b.clone().detach().T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
    } for b in batch]
    return collate(adapted_batch)

def style100_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[0].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'lengths': b[1],
        'style_name': b[2],
        'motion_t': b[3],
        'motion_s': b[4],
        'motion_s_name': b[5]
    } for b in batch]
    return collate(adapted_batch)

def style100sample_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[0].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'lengths': b[1],
        'style_name': b[2],
        'motion_s': b[3],
        'motion_s_name': b[4]
    } for b in batch]
    return collate(adapted_batch)

def permo_onlytext_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[1].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[0], #b[0]['caption']
        'lengths': b[2],
    } for b in batch]
    return collate(adapted_batch)