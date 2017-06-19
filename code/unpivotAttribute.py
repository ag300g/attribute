# -*- coding: utf-8 -*-
import pandas as pd
def unpivotAttributes(df, maps):
    # add back in the product keys before unpivot
    (skuKey, skuName) = ('ProductKey', 'ProductDesc')
    df = pd.merge(maps[skuKey], df, how='outer', on=skuName)
    df = df.set_index([skuKey, skuName], drop=True, inplace=False)

    # unpivot and reset the index
    df = df.stack()
    df = pd.DataFrame(df)
    df.reset_index(inplace=True)

    # rename columns to enable joins
    mappings = {'AttributeKey': 'AttributeDesc', 'AttributeValueKey': 'AttributeValueDesc'}
    colnames = [skuKey, skuName] + list(mappings.values())
    df.columns = colnames

    # add back in the attribute and attribute value keys after unpivot
    for key, val in mappings.items():
        df = pd.merge(maps[key], df, how='outer', on=val)
        df = df.dropna()

    return df
