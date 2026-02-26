Modified IDL Transformation in RQ4
--------------------

In RQ4, we investigate whether an MIA classifier trained on one model can effectively transfer to other target models. Our experiments show that Code Llama 7B achieves the best overall performance when serving as the target model. Moreover, we observe that certain transfer cases (e.g., Deepseek-Coder 7B -> Code Llama 7B on HumanEval) yield higher AUC than their corresponding non-transfer cases (e.g., Deepseek-Coder 7B -> Deepseek-Coder 7B), which may be due to some perturbations being more favorable to specific victim models.

To examine whether this favorable performance stems from the perturbations employed, we take the IDL transformation as an example and revise it by replacing the simple loop body with more sophisticated code snippets. The revised IDL remains semantics-preserving, as all conditions are false and loop bodies are never executed. A comparison of the IDL transformation before and after the modification is shown below.

The loop body used in the original IDL transformation is as follows:

- case 1: `print("Debug: Entering loop")`
- case 2: `pass`
- case 3: `variable_definition`

The updated loop body is presented as follows:

Case 1:

```python
# complex body A: list comp + helper + safe conversion
_tmp = [i*i for i in range(3)]
def _helper(arr):
    return ''.join(str(e) for e in arr)
_s =_ helper(_tmp)
try:
    _ = int(_ s)  # will typically fail and be caught
except Exception:
    pass
```
Case 2:
```python
# complex body B: local class + lambda + comprehension,
class _C:
    def __init__(self):
        self._data = [None] * 4
    def f(self):
        return sum(1 for v in self._data if v is None)
_c = _C()
_lst = [str(i) for i in range(2)]
_rev = (lambda z: z[::-1])(_lst)
if False:
    print('dead')
```
Case 3:
```python
# unused-var body A: several unused complex assignments
_a = {const_a}
_b, _c = (_a, len(str(_a)))
_d = {i: str(i) for i in range(3)}
_e = [v for v in _d.values()]
_f = (_e, _b)
```

We then compared the performance changes of our method on two target models, Code Llama 7B and WizardCoder 7B, before and after the IDL modification. These two were chosen because they represent the best- and worst-performing target models on the APPS dataset. As shown in the results (Fig.11 in our paper), performance declined on Code Llama 7B but improved significantly on WizardCoder 7B. These findings indicate that while our method is effective across all victim models, perturbations influence them differently, underscoring adaptive model-specific perturbation optimization as a promising direction for future research.
