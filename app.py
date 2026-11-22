"""BOJ simulator: IS–LM is the main model; Taylor Rule is a separate extension."""
from dataclasses import replace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from textbook_model import (CourseParameters, solve_course, is_nominal_rate,
                            lm_rate, pc_price, pc_inflation, taylor_rate, apply_shock)

st.set_page_config(page_title='BOJ | Macroeconomic Policy Simulator', page_icon='🇯🇵', layout='wide')
st.title('Bank of Japan | Macroeconomic Policy Simulator')
st.caption('IS–LM main framework · Taylor Rule comparison based on Kurlat Chapters 14–15 and Kazakova Lecture 8')

page=st.sidebar.radio('Navigation', ['Main Model — IS–LM', 'Extension — Taylor Rule'], index=0)
main_model=page.startswith('Main')
st.sidebar.caption('The IS–LM framework is the main project model. Taylor Rule is a separate comparison.')
with st.sidebar:
    st.header('Economic shock')
    if main_model:
        options=['No shock','Increase money supply','Higher expected future inflation',
                 'Lower discount factor (impatience)','Higher government spending']
    else:
        options=['No shock','Higher expected future inflation','Lower discount factor (impatience)',
                 'Higher government spending','Cost-push shock',
                 'Policy-rate shock','Change inflation expectations (PC)']
    shock=st.selectbox('Shock type',options)
    size=st.slider('Shock magnitude',-40.0,40.0,10.0,1.0,
        help='Money supply and beta shocks: percent changes; inflation and policy-rate shocks: percentage points; government purchases: output units.')
    st.header('IS parameters')
    beta=st.slider('Discount factor (β)',0.2,0.99,0.5,0.01)
    pi_future=st.slider('Expected future inflation (π, %)',-10.,20.,0.,.5)/100
    g=st.slider('Government purchases (G)',0.,.5,0.,.01)
    if main_model:
        st.header('LM / Price parameters')
        lam=st.slider('Money demand parameter (λ)',.2,3.,1.,.1)
        ms=st.slider('Money supply (Mˢ)',.2,3.,1.,.05)
        p1=st.slider('Reference price level (p₁)',.5,2.,1.,.05)
        y_n=st.number_input('Reference output (Yᴺ)',value=.68,step=.05)
        price_slope=st.slider('Price-level PC slope',0.,1.,.25,.05)
        r_n=0.; pi_exp=.02; pi_target=.02; pc_a=.05; ta=.5; tb=.5
    else:
        # Money supply is not chosen independently in this operating regime.
        lam=1.; ms=1.; p1=1.; price_slope=.25
        st.header('Phillips Curve / Taylor Rule')
        y_n=st.number_input('Natural output (Yᴺ)',value=.68,step=.05)
        r_n=st.slider('Natural real interest rate (rᴺ, %)',-5.,8.,0.,.25)/100
        pi_exp=st.slider('Expected inflation E(π₁), %',-2.,8.,2.,.25)/100
        pi_target=st.slider('Inflation target (π*, %)',0.,5.,2.,.25)/100
        pc_a=st.slider('Inflation Phillips Curve slope (a_PC)',0.,.3,.05,.01)
        ta=st.slider('Taylor inflation response (a)',0.,3.,.5,.1)
        tb=st.slider('Taylor output response (b)',0.,3.,.5,.1)

p=CourseParameters(beta=beta, expected_future_pi=pi_future, lam=lam,p1=p1,
    money_supply=ms, g=g,y_n=y_n,r_n=r_n,pi_expected=pi_exp,pi_target=pi_target,
    pc_a=pc_a,taylor_a=ta,taylor_b=tb,price_pc_slope=price_slope)
regime='Money Supply / IS-LM' if main_model else 'Taylor Rule / IS-PC'
try:
    shocked_p=apply_shock(p,shock,size)
    before=solve_course(p,regime)
    after=solve_course(shocked_p,regime)
except (ValueError,ZeroDivisionError,OverflowError) as exc:
    st.error(f'No equilibrium found for these inputs: {exc}')
    st.stop()

st.header('IS–LM Model' if main_model else 'Taylor Rule — Policy Comparison')
if main_model:
    st.write('Goods-market equilibrium, money-market equilibrium and price adjustment. The price-level PC and LM are linked, so changes in the price level also change the LM relationship.')
else:
    st.write('A secondary comparison using an interest-rate rule and the expectations-augmented inflation Phillips Curve. This is separate from the main IS–LM model.')

m1,m2,m3,m4=st.columns(4)
m1.metric('Output (Y₁)',f"{after['Y1']:.3f}",f"{after['Y1']-before['Y1']:+.3f}")
m2.metric('Nominal interest rate (i)',f"{after['i']*100:.2f}%",f"{(after['i']-before['i'])*100:+.2f} pp")
if main_model:
    m3.metric('Price level (p₁)',f"{after['price_level']:.3f}",f"{after['price_level']-before['price_level']:+.3f}")
else:
    m3.metric('Inflation (π₁)',f"{after['pi1']*100:.2f}%",f"{(after['pi1']-before['pi1'])*100:+.2f} pp")
m4.metric('Output gap (Y₁−Yᴺ)',f"{after['output_gap']:+.3f}")

def y_range():
    low=max(.01,min(before['Y1'],after['Y1'],p.y_n)*.55)
    high=max(before['Y1'],after['Y1'],p.y_n)*1.55
    return np.linspace(low,max(high,low+.25),450)

def draw_is():
    ys=y_range(); fig,ax=plt.subplots(figsize=(7,4.6),layout='constrained')
    for par,tag,style in [(p,'Before','--'),(shocked_p,'After','-')]:
        ax.plot(ys,100*is_nominal_rate(ys,par),style,c='#2563eb',lw=2,label=f'IS — {tag}')
        policy=lm_rate(ys,par) if main_model else taylor_rate(ys,par)
        ax.plot(ys,100*policy,style,c='#dc2626',lw=2,label=f'{"LM" if main_model else "Taylor Rule"} — {tag}')
    for sol,name,col in [(before,'E₀','#64748b'),(after,'E₁','#16a34a')]:
        ax.scatter(sol['Y1'],sol['i']*100,s=75,c=col,zorder=5)
        ax.annotate(name,(sol['Y1'],sol['i']*100),xytext=(7,7),textcoords='offset points')
    rates=[before['i']*100,after['i']*100]
    lim=max(15,abs(rates[1]-rates[0])*3)
    ax.set_ylim(min(rates)-lim,max(rates)+lim)
    ax.set_xlabel('Current output (Y₁)');ax.set_ylabel('Nominal interest rate (i, %)')
    ax.set_title('IS–LM Equilibrium' if main_model else 'IS and Taylor Rule')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    return fig

def draw_pc():
    ys=y_range();fig,ax=plt.subplots(figsize=(7,4.6),layout='constrained')
    for par,tag,style in [(p,'Before','--'),(shocked_p,'After','-')]:
        values=pc_price(ys,par) if main_model else 100*pc_inflation(ys,par)
        ax.plot(ys,values,style,lw=2,c='#0d9488',label=f'PC — {tag}')
    for sol,name,col in [(before,'E₀','#64748b'),(after,'E₁','#16a34a')]:
        value=sol['price_level'] if main_model else 100*sol['pi1']
        ax.scatter(sol['Y1'],value,s=75,c=col,zorder=5)
        ax.annotate(name,(sol['Y1'],value),xytext=(7,7),textcoords='offset points')
    ax.axvline(p.y_n,ls=':',c='gray',lw=1,label='Yᴺ')
    ax.set_xlabel('Current output (Y₁)')
    ax.set_ylabel('Price level (p₁)' if main_model else 'Inflation (π₁, %)')
    ax.set_title('Price-Level Phillips Curve' if main_model else 'Expectations-Augmented Phillips Curve')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
    return fig

tab_graph,tab_equations,tab_export=st.tabs(['Equilibrium diagrams','Equations & interpretation','Results & CSV'])
with tab_graph:
    l,r=st.columns(2)
    with l:
        fig=draw_is();st.pyplot(fig);plt.close(fig)
    with r:
        fig=draw_pc();st.pyplot(fig);plt.close(fig)
with tab_equations:
    st.subheader('Model equations')
    st.latex(r'\mathrm{IS:}\quad Y_1=\frac{2+\beta}{16\beta(1+i-\pi)^2}+G')
    if main_model:
        st.latex(r'\mathrm{PC\ (price\ level):}\quad p_1(Y_1)=\bar p_1\exp\!\left(\gamma_p(Y_1-Y^N)\right)')
        st.latex(r'\mathrm{LM:}\quad i=\frac{\lambda\,p_1(Y_1)}{M^S}Y_1-1')
        st.write('Price adjustment is linked to money-market equilibrium: when p₁ changes, real money balances and the LM relationship change as well.')
        st.write('With fixed nominal money supply, a higher price level reduces real balances, shifting LM upward. We solve the IS equation together with this price-adjusted LM relation.')
    else:
        st.latex(r'\mathrm{PC:}\quad \pi_1-E(\pi_1)=a_{PC}(Y_1-Y^N)+u')
        st.latex(r'\mathrm{Taylor:}\quad i=r^N+\pi_1+a(\pi_1-\pi^*)+b(Y_1-Y^N)+m')
        st.write('The Phillips Curve and Taylor Rule determine inflation and the policy rate in this alternative framework.')
        st.write('The Taylor Rule is an alternative interest-rate setting assumption. Money supply is not an independent policy input in this comparison.')
    st.write('Notation: π represents expected inflation between periods 1 and 2; π₁ denotes current inflation.')
with tab_export:
    rows=[]
    for name,sol in [('Before',before),('After',after)]:
        rows.append({'Scenario':name,'Y1':sol['Y1'],'Nominal interest rate (%)':sol['i']*100,
            'Price level p1':sol['price_level'],'Inflation (%)':sol['pi1']*100,'Output gap':sol['output_gap']})
    frame=pd.DataFrame(rows)
    st.dataframe(frame,use_container_width=True)
    st.download_button('Download results as CSV',frame.to_csv(index=False),file_name='boj_simulation.csv',mime='text/csv')
    st.caption(f"Model equation residuals: IS {after['is_residual']:.2e}; policy {after['policy_residual']:.2e}.")
