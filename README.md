# Bank of Japan — Macroeconomic Policy Simulator (V5)

## Website structure

- **Main Model — IS–LM:** nonlinear IS, money-market clearing, price-level Phillips Curve, and equilibrium shock analysis.
- **Extension — Taylor Rule:** comparison of an interest-rate rule and inflation Phillips Curve.

## Changing model parameters

Adjust the parameters directly in the website sidebar. To change the default values or add more controls, edit the `CourseParameters` defaults in `textbook_model.py` and the corresponding controls in `app.py`. The solver and plotting functions are separate, so the Python model can be updated without redesigning the website.

## Model specification notes

The nonlinear IS and LM relationships follow Kazakova Lecture 8, pp. 20–28. The Taylor Rule and inflation Phillips Curve refer to Kurlat Chapters 14–15. The price-level Phillips Curve uses an explicit reduced-form specification `p1(Y1)=p_anchor*exp(gamma_p*(Y1-Y_n))`; the government-purchases, cost-push and policy-wedge terms are additional model choices. The IS–LM and Taylor Rule pages use alternative monetary-policy regimes, not two simultaneous independent policy rules.

Here `π` in IS is expected inflation between periods, while `π₁` in the inflation Phillips Curve is current inflation. Internally, rates are decimal fractions; charts show percentages.

## Run in VS Code

```powershell
& "C:\Users\ryan1\anaconda3\python.exe" -m pip install -r requirements.txt
& "C:\Users\ryan1\anaconda3\python.exe" -m streamlit run app.py
```

Keep `app.py`, `textbook_model.py`, `model.py` and `requirements.txt` in the same folder.
