import jieba
```
lcut can result in a list 
```
def tokenise(df, colname=SKU_NAME):
    cuts = {
        'cut_search': lambda s: jieba.lcut_for_search(s),
        'cut_full': lambda s: jieba.lcut(s, cut_all=True),
        'cut_part': lambda s: jieba.lcut(s, cut_all=False)
    }
    for cut_name, cut in cuts.items():
        df[cut_name]=df[colname].map(cut)
    return df
    
