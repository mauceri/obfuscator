"""Wrapper client : tokenize+permute à l'envoi, dépermute+detokenize à la réception.

La permutation du vocabulaire est la clé qui reste côté client : le serveur
ne voit jamais que des IDs permutés, et sa table d'embedding obfusquée est
réindexée pour que la ligne `permutation[t]` porte les données du token clair
`t` (cf. `embedding_obfuscation.py`). Les logits qu'il renvoie sont donc
eux aussi dans l'espace permuté, d'où le `unpermute` au retour.
"""


class ClientCodec:
    def __init__(self, permutation, unpermute, tokenizer):
        # JSON transforme les clés d'objet en chaînes : les tables relues depuis
        # `obfuscation_keys.json` arrivent avec des clés `str`, et un `encode()`
        # lèverait KeyError. La coercition est faite ici, une fois, plutôt que
        # laissée à chaque appelant.
        self.permutation = {int(k): int(v) for k, v in permutation.items()}
        self.unpermute = {int(k): int(v) for k, v in unpermute.items()}
        self.tokenizer = tokenizer

    def encode(self, text):
        clear_ids = self.tokenizer.encode(text)
        return [self.permutation[i] for i in clear_ids]

    def decode(self, permuted_ids):
        clear_ids = [self.unpermute[i] for i in permuted_ids]
        return self.tokenizer.decode(clear_ids)
