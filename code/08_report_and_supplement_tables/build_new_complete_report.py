from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "08_新完整报告"
ASSET_DIR = REPORT_DIR / "report_assets"
TEX_PATH = REPORT_DIR / "PM25_integrated_full_report.tex"
PDF_PATH = REPORT_DIR / "PM25_integrated_full_report.pdf"


def copy_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    sources = [
        ROOT / "05_图表/report_assets/desc_summary_panels.png",
        ROOT / "05_图表/report_assets/bn_summary_panels.png",
        ROOT / "05_图表/report_assets/rf_summary_panels.jpg",
        ROOT / "05_图表/report_assets/lasso_summary_panels.png",
        ROOT / "05_图表/report_assets/mlr_diagnostic_panels.png",
        ROOT / "05_图表/report_assets/gam_diagnostic_importance_panels_clean.png",
        ROOT.parent / "概率论/预测模型/figures/best_ensemble_observed_vs_predicted.png",
        ROOT.parent / "概率论/预测模型/figures/daily_prediction_timeseries.png",
        ROOT.parent / "概率论/预测模型/figures/trend_horizon_rmse.png",
    ]
    for src in sources:
        if src.exists():
            shutil.copy2(src, ASSET_DIR / src.name)


def write_tex() -> None:
    tex = r"""
\documentclass[11pt]{article}
\usepackage[a4paper,margin=0.78in]{geometry}
\usepackage{setspace}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{caption}
\usepackage{microtype}
\usepackage{placeins}
\usepackage{enumitem}
\usepackage{titlesec}
\usepackage{hyperref}

\setstretch{1.08}
\setlength{\parskip}{0.28em}
\setlength{\parindent}{0pt}
\setlength{\textfloatsep}{9pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\setlength{\abovedisplayskip}{7pt}
\setlength{\belowdisplayskip}{7pt}
\setlist{noitemsep,topsep=2pt,leftmargin=1.4em}
\graphicspath{{report_assets/}}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\titlespacing*{\section}{0pt}{1.6ex plus 0.3ex}{0.9ex}
\titlespacing*{\subsection}{0pt}{1.1ex plus 0.2ex}{0.5ex}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\title{Integrated PM$_{2.5}$ Analysis and Forecasting Report\\\large Beijing Daily Air Pollution: Mechanism, Nonlinearity, and Prediction}
\author{Ban Jingyang, Chen Anyi, Chen Lichong, Feng Shuo, Lian Yuehan, and Wu Ziqi}
\date{""" + date.today().isoformat() + r"""}

\begin{document}
\maketitle
\tableofcontents
\newpage

\begin{abstract}
This report rewrites the PM$_{2.5}$ analysis into one unified experimental narrative organized around three modules: problem discovery, key-variable identification, and mechanism explanation. The central mechanism question is: which meteorological factors, gaseous pollutants, and persistence terms jointly explain daily PM$_{2.5}$ in Beijing, and where do their effects become strong? Module 1 combines descriptive statistics and Bayesian-network risk analysis to ask what PM$_{2.5}$ looks like and under what states high pollution is more likely. Module 2 combines random forest and LASSO to identify variables that remain important under nonlinear prediction and regularized screening. Module 3 combines full OLS and GAM to explain direction, significance, and nonlinear form. A separate forecasting extension then evaluates whether lagged information can predict future PM$_{2.5}$. The most important correction is that Step 5 is unified around the full OLS model containing meteorology, gaseous pollutants, lagged PM$_{2.5}$, and seasonal controls. This model reaches $R^2=0.7698$, while the weather-only OLS reaches only $R^2=0.4744$ and is therefore treated as a forecasting-branch baseline rather than the main mechanism model.
\end{abstract}

\section{Research Question and Unified Workflow}
The main research question is:
\begin{quote}
How do meteorological conditions, gaseous pollutants, and short-term pollution persistence jointly influence daily PM$_{2.5}$ in Beijing, and can the same information support short-term forecasting?
\end{quote}

The analysis is divided into two related but distinct parts. The first part is the mechanism workflow: it explains daily PM$_{2.5}$ using descriptive statistics, Bayesian networks, random forest, LASSO, OLS, and GAM. The second part is a forecasting extension: it asks whether previous-day and historical information can predict future PM$_{2.5}$. This separation is essential because explanation and forecasting use different information sets.

The final workflow is therefore organized as:
\begin{enumerate}
\item Module 1, problem discovery: Step 1 descriptive statistics and Step 2 Bayesian network. This module answers what distributional features PM$_{2.5}$ has and under which concrete states high pollution is more likely.
\item Module 2, key-variable identification: Step 3 random forest and Step 4 LASSO. This module answers which variables repeatedly matter in nonlinear prediction and regularized screening.
\item Module 3, mechanism explanation: Step 5 full OLS and Step 6 GAM. This module answers the direction, significance, and nonlinear shape of the key effects.
\item Forecasting extension: lagged-feature prediction models evaluate whether historical information can predict future PM$_{2.5}$.
\end{enumerate}

\section{Data and Variable Construction}
The daily dataset contains 1,462 observations; after constructing lagged PM$_{2.5}$, the main regression sample contains 1,461 valid observations. The dependent variable is daily PM$_{2.5}$ concentration. Explanatory variables include temperature, humidity, wind speed, precipitation, CO, NO$_2$, SO$_2$, O$_3$, lagged PM$_{2.5}$, month, and seasonal indicators. PM$_{10}$ is retained for descriptive and Bayesian-network analysis as particulate background, but it is excluded from the main OLS/GAM mechanism model to avoid excessive overlap with PM$_{2.5}$.

The Bayesian-network state variables are based on tertile cutpoints. High PM$_{2.5}$ is defined as PM$_{2.5}>32.24$ $\mu$g/m$^3$. Important cutpoints include low wind speed $\leq1.65$ m/s, high wind speed $>2.40$ m/s, high humidity $>67.29\%$, high CO $>0.54$ mg/m$^3$, and high NO$_2>25.92$ $\mu$g/m$^3$.

\section{Module 1: Discovering the Problem}
This module combines Step 1 descriptive statistics and Step 2 Bayesian-network risk analysis. It answers two foundational questions: what distributional features PM$_{2.5}$ has, and under which meteorological or pollutant states high pollution is more likely.

\subsection{Step 1: Descriptive Statistics}
Descriptive statistics provide the empirical foundation for the later models. PM$_{2.5}$ is right-skewed: its mean is 30.10 $\mu$g/m$^3$, median is 22.71, and maximum is 164.34. The mean being higher than the median indicates that a small number of severe pollution days raise the average. Lagged PM$_{2.5}$ has almost the same distribution as current PM$_{2.5}$, showing strong short-term persistence. Precipitation has a median of zero and a high maximum, so it is better interpreted both as a rain/no-rain state and as a continuous amount.

\begin{table}[!htbp]
\centering
\caption{Descriptive Statistics of Key Variables}
\small
\begin{tabular}{lrrrrrr}
\toprule
Variable & $n$ & Mean & SD & Min & Median & Max \\
\midrule
PM$_{2.5}$ & 1,462 & 30.10 & 25.72 & 1.63 & 22.71 & 164.34 \\
Temperature & 1,462 & 12.26 & 11.89 & -16.12 & 13.63 & 33.27 \\
Humidity & 1,462 & 58.28 & 17.37 & 15.24 & 58.17 & 96.00 \\
Wind speed & 1,462 & 2.23 & 1.02 & 0.43 & 2.01 & 9.71 \\
Precipitation & 1,462 & 2.21 & 7.73 & 0.00 & 0.00 & 97.29 \\
CO & 1,462 & 0.49 & 0.20 & 0.11 & 0.46 & 1.45 \\
NO$_2$ & 1,462 & 23.14 & 12.43 & 2.44 & 20.10 & 66.21 \\
O$_3$ & 1,462 & 62.07 & 32.87 & 5.10 & 58.24 & 192.21 \\
PM$_{10}$ & 1,462 & 57.53 & 50.75 & 4.16 & 46.28 & 696.56 \\
Lagged PM$_{2.5}$ & 1,461 & 30.11 & 25.72 & 1.63 & 22.75 & 164.34 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\textwidth]{desc_summary_panels.png}
\caption{Descriptive plots: time series, PM$_{2.5}$ histogram, boxplots, and correlation heatmap.}
\end{figure}

The descriptive stage motivates two later choices. First, high-pollution probability should be studied directly because PM$_{2.5}$ has a long right tail. Second, lagged PM$_{2.5}$, CO, NO$_2$, and PM$_{10}$ should be carefully tracked because they are repeatedly associated with PM$_{2.5}$.
\FloatBarrier

\subsection{Step 2: Bayesian Network and Concrete Risk States}
The Bayesian network asks how the probability of high PM$_{2.5}$ changes under specific variable states. The baseline probability of high PM$_{2.5}$ is 33.38\%. The network should be interpreted as a probabilistic association model, not a causal diagram.

\begin{equation}
P(Y,X_1,\ldots,X_p)=P(Y)\prod_{j=1}^{p}P(X_j\mid Y), \quad
P(Y=\mathrm{High}\mid E)=\frac{P(Y=\mathrm{High})P(E\mid Y=\mathrm{High})}{\sum_y P(Y=y)P(E\mid Y=y)} .
\end{equation}

\subsubsection{Wind Speed}
Wind speed must be reported by concrete ranges. Low wind speed, defined as $\leq1.65$ m/s, raises the high-pollution probability to 41.55\%, which is 8.17 percentage points above baseline. High wind speed, defined as $>2.40$ m/s, lowers it to 24.29\%, which is 9.09 percentage points below baseline. The observed mean PM$_{2.5}$ also falls from 36.88 under low wind speed to 23.70 under high wind speed. Therefore, the correct statement is not just ``wind speed affects PM$_{2.5}$'', but that low wind speed is a higher-risk stagnation state and high wind speed is a lower-risk dispersion state.

\subsubsection{CO and NO$_2$}
CO is the strongest gaseous-pollutant signal. When CO is high ($>0.54$ mg/m$^3$), the high-pollution probability reaches 75.36\%, 41.98 percentage points above baseline. When CO is low ($\leq0.38$ mg/m$^3$), the probability is only 3.27\%. NO$_2$ shows a similar but weaker pattern: high NO$_2$ ($>25.92$ $\mu$g/m$^3$) corresponds to a 64.77\% high-pollution probability, while low NO$_2$ ($\leq15.91$ $\mu$g/m$^3$) corresponds to 10.59\%.

\subsubsection{Humidity, Temperature, O$_3$, and Precipitation}
Humidity is not best described as a single linear slope. Low humidity ($\leq49.31\%$) corresponds to 19.76\% high-pollution probability, while medium and high humidity correspond to about 40\%. Temperature is also non-monotonic: medium temperature (5.47--20.63$^\circ$C) has the highest BN probability, 42.45\%, while high temperature has only 21.59\%. O$_3$ shows a U-shaped state pattern: low O$_3$ has 43.67\% high-pollution probability, medium O$_3$ has 23.88\%, and high O$_3$ has 32.59\%. Rain has a smaller contrast: no rain has 35.16\% probability and rain has 31.44\%.

\begin{table}[!htbp]
\centering
\caption{Concrete State Risks from the Bayesian Network}
\small
\begin{tabular}{llrrr}
\toprule
Variable state & Concrete definition & BN $P_H$ & Change & Mean PM$_{2.5}$ \\
\midrule
Low wind & $\leq1.65$ m/s & 41.55\% & +8.17 pp & 36.88 \\
High wind & $>2.40$ m/s & 24.29\% & -9.09 pp & 23.70 \\
High CO & $>0.54$ mg/m$^3$ & 75.36\% & +41.98 pp & 52.95 \\
High NO$_2$ & $>25.92$ $\mu$g/m$^3$ & 64.77\% & +31.39 pp & 49.45 \\
High humidity & $>67.29\%$ & 40.29\% & +6.91 pp & 36.35 \\
No rain & Precipitation = 0 & 35.16\% & +1.78 pp & 31.08 \\
High PM$_{10}$ & $>62.30$ $\mu$g/m$^3$ & 84.11\% & +50.74 pp & 55.95 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\textwidth]{bn_summary_panels.png}
\caption{Bayesian-network structure, state probabilities, sensitivity ranking, and key scenarios.}
\end{figure}
\FloatBarrier

\section{Module 2: Identifying Key Variables}
This module combines Step 3 random forest and Step 4 LASSO. It answers which variables repeatedly appear important when the model is allowed to be nonlinear and when the linear predictor set is regularized under collinearity.

\subsection{Step 3: Random Forest Regression}
Random forest provides a nonlinear predictive benchmark and identifies variables that matter for continuous PM$_{2.5}$ prediction. The model uses 1,168 training rows, 293 test rows, and 14 predictors. Its test RMSE is 11.64, MAE is 7.13, test $R^2$ is 0.7787, and out-of-bag $R^2$ is 0.7876.

\begin{equation}
\widehat f_{RF}(\mathbf{x})=\frac{1}{B}\sum_{b=1}^{B}T_b(\mathbf{x})
\end{equation}

The random forest importance ranking strongly supports the BN result. CO accounts for 57.77\% of feature importance, lagged PM$_{2.5}$ for 18.80\%, and NO$_2$ for 8.46\%. Together they explain about 85\% of total impurity-based importance.

\begin{table}[!htbp]
\centering
\caption{Random Forest Performance and Main Importance Results}
\small
\begin{tabular}{lr@{\qquad}lr}
\toprule
Metric & Value & Feature & Importance share \\
\midrule
Test RMSE & 11.64 & CO & 57.77\% \\
Test MAE & 7.13 & Lagged PM$_{2.5}$ & 18.80\% \\
Test $R^2$ & 0.7787 & NO$_2$ & 8.46\% \\
OOB $R^2$ & 0.7876 & Humidity & 2.78\% \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\textwidth]{rf_summary_panels.jpg}
\caption{Random-forest feature importance, observed-versus-predicted plot, residuals, and residual distribution.}
\end{figure}
\FloatBarrier

\subsection{Step 4: LASSO Variable Selection}
LASSO is used as a regularized linear screening method under collinearity. It solves
\begin{equation}
\widehat{\boldsymbol\beta}=\arg\min_{\beta_0,\boldsymbol\beta}\left\{\frac{1}{n}\sum_{i=1}^{n}(y_i-\beta_0-\mathbf{x}_i^\top\boldsymbol\beta)^2+\lambda\sum_j|\beta_j|\right\}.
\end{equation}
Cross-validation selects $\alpha_{\min}=0.0517$, while the one-standard-error alternative is $\alpha_{1se}=2.5235$. The minimum-error solution is used because it preserves more information for comparison with OLS and GAM.

\begin{table}[!htbp]
\centering
\caption{Selected Standardized LASSO Coefficients}
\small
\begin{tabular}{lrl@{\qquad}lrl}
\toprule
Variable & Coef. & Direction & Variable & Coef. & Direction \\
\midrule
CO & 14.300 & Positive & Wind speed & 1.112 & Positive \\
NO$_2$ & 7.946 & Positive & Precipitation & -0.732 & Negative \\
Lagged PM$_{2.5}$ & 6.952 & Positive & Month & -1.454 & Negative \\
Spring & 2.455 & Positive & Winter & -1.513 & Negative \\
O$_3$ & 1.734 & Positive & SO$_2$ & -2.058 & Negative \\
\bottomrule
\end{tabular}
\end{table}

The key result is again the same: CO, NO$_2$, and lagged PM$_{2.5}$ remain the leading positive terms. Smaller coefficients should be treated cautiously because LASSO can select among correlated predictors and shrink unstable effects.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\textwidth]{lasso_summary_panels.png}
\caption{LASSO cross-validation, coefficient path, and selected standardized coefficients.}
\end{figure}
\FloatBarrier

\section{Module 3: Explaining the Influence Mechanism}
This module combines Step 5 full OLS and Step 6 GAM. It answers how key variables affect PM$_{2.5}$ in terms of conditional direction, statistical significance, and nonlinear partial-effect shape.

\subsection{Step 5: Unified Full OLS Model}
\subsubsection{Why the Full Model Must Be Used}
The most important revision in this report is the Step 5 unification. The formal mechanism model is the full OLS:
\begin{equation}
\begin{aligned}
\mathrm{PM}_{2.5,t}=&\beta_0+\beta_1\mathrm{Temperature}_t+\beta_2\mathrm{Humidity}_t+\beta_3\mathrm{WindSpeed}_t+\beta_4\mathrm{Precipitation}_t\\
&+\beta_5\mathrm{CO}_t+\beta_6\mathrm{NO}_{2,t}+\beta_7\mathrm{SO}_{2,t}+\beta_8\mathrm{O}_{3,t}+\beta_9\mathrm{LagPM}_{2.5,t}\\
&+\beta_{10}\mathrm{Spring}_t+\beta_{11}\mathrm{Summer}_t+\beta_{12}\mathrm{Autumn}_t+\varepsilon_t .
\end{aligned}
\end{equation}

This model contains meteorology, gaseous pollutants, lagged PM$_{2.5}$, and seasonal controls. It must replace the weather-only OLS as the main Step 5 result. The reason is numerical and conceptual: the full OLS has $R^2=0.7698$, while the weather-only OLS has $R^2=0.4744$. Adding gaseous pollutants increases $R^2$ by about 0.2954, so a paper about meteorology and gaseous pollutants jointly influencing PM$_{2.5}$ cannot omit the gaseous pollutants from Step 5.

\begin{table}[!htbp]
\centering
\caption{Step 5 Model Comparison}
\small
\begin{tabular}{lrrrr}
\toprule
Model & $R^2$ & Adjusted $R^2$ & RMSE & Intended use \\
\midrule
Full OLS: meteorology + gases + lag & 0.7698 & 0.7679 & 12.34 & Mechanism Step 5 \\
Weather-only OLS: meteorology + lag & 0.4744 & 0.4715 & 18.64 & Forecasting branch only \\
\bottomrule
\end{tabular}
\end{table}

\subsubsection{Coefficient Interpretation}
In the full OLS, CO, NO$_2$, and lagged PM$_{2.5}$ are the most important positive terms. CO has a raw coefficient of 71.96 and a standardized coefficient of 0.565. NO$_2$ has a raw coefficient of 0.635 and a standardized coefficient of 0.307. Lagged PM$_{2.5}$ has a raw coefficient of 0.271 and a standardized coefficient of 0.271. This means that pollutant co-movement and short-term persistence dominate the linear mechanism model.

\begin{table}[!htbp]
\centering
\caption{Main Coefficients in the Unified Full OLS}
\small
\begin{tabular}{lrrrl}
\toprule
Variable & Raw coef. & Std. coef. & Approx. $p$ & Interpretation \\
\midrule
CO & 71.96 & 0.565 & $<0.001$ & strongest positive association \\
NO$_2$ & 0.635 & 0.307 & $<0.001$ & stable gaseous-pollutant effect \\
Lagged PM$_{2.5}$ & 0.271 & 0.271 & $<0.001$ & short-term persistence \\
Spring & 9.986 & 0.169 & $<0.001$ & seasonal shift vs. winter \\
O$_3$ & 0.066 & 0.084 & 0.001 & positive conditional association \\
Wind speed & 1.047 & 0.042 & 0.021 & small conditional linear term \\
Temperature & -0.147 & -0.068 & 0.041 & weak negative after controls \\
Precipitation & -0.092 & -0.028 & 0.063 & weak removal direction \\
\bottomrule
\end{tabular}
\end{table}

The wind-speed coefficient deserves special care. In the BN state analysis, low wind speed is a high-risk state and high wind speed is a low-risk state. In full OLS, after controlling for CO, NO$_2$, SO$_2$, O$_3$, lagged PM$_{2.5}$, and season, wind speed has a small positive conditional coefficient. This does not mean that wind always increases PM$_{2.5}$. It means the linear coefficient is conditional on correlated variables and should be read together with BN and GAM.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\textwidth]{mlr_diagnostic_panels.png}
\caption{Full OLS diagnostics: observed versus predicted values, residuals, residual distribution, and Q-Q plot.}
\end{figure}
\FloatBarrier

\subsection{Step 6: GAM Nonlinear Diagnostics}
GAM extends the OLS framework by replacing selected linear terms with smooth functions:
\begin{equation}
\mathrm{PM}_{2.5,t}=\beta_0+s_1(\mathrm{WindSpeed}_t)+s_2(\mathrm{Humidity}_t)+s_3(\mathrm{NO}_{2,t})+s_4(\mathrm{CO}_t)+s_5(\mathrm{O}_{3,t})+\mathbf{z}_t^\top\boldsymbol\gamma+\varepsilon_t .
\end{equation}
The GAM reaches $R^2=0.7930$, adjusted $R^2=0.7881$, RMSE=11.70, and AIC=11,402.51. Its AIC is 183.89 lower than the linear baseline, showing that nonlinear structure adds real explanatory value.

\begin{table}[!htbp]
\centering
\caption{GAM Smooth-Term Contributions and Concrete Effect Ranges}
\small
\begin{tabular}{lrrl}
\toprule
Smooth term & Drop dev. & $p$ & Concrete nonlinear interpretation \\
\midrule
CO & 0.0692 & $<0.001$ & positive and significant from 0.54--1.08 mg/m$^3$ \\
NO$_2$ & 0.0544 & $<0.001$ & positive and significant from 25.43--58.67 $\mu$g/m$^3$ \\
O$_3$ & 0.0187 & $<0.001$ & negative at 8.65--37.46, positive at 46.22--155.20 \\
Humidity & 0.0072 & $<0.001$ & positive at 73.82--91.28\%, negative at 52.29--66.84\% \\
Wind speed & 0.0020 & 0.025 & negative around 1.79--1.94 and 2.25--2.37 m/s \\
\bottomrule
\end{tabular}
\end{table}

CO and NO$_2$ have the clearest threshold agreement between BN and GAM. BN says high CO begins above about 0.54 mg/m$^3$ and high NO$_2$ above about 25.92 $\mu$g/m$^3$; GAM finds significant positive ranges starting near the same values. Humidity and wind speed are more complex, so they should be described as nonlinear meteorological modifiers rather than simple monotone drivers.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\textwidth]{gam_diagnostic_importance_panels_clean.png}
\caption{GAM diagnostics and smooth-term ranking.}
\end{figure}
\FloatBarrier

\section{Forecasting Extension: Predicting PM$_{2.5}$}
The forecasting experiment is included as a separate extension. It should not be merged with Step 5, because the forecasting question is different: it asks whether previous-day and historical information can predict future PM$_{2.5}$. The feature set includes lagged PM$_{2.5}$, AQI, meteorology, gaseous pollutants, rolling means, rolling standard deviations, slopes, seasonality, and wind-direction transforms.

\subsection{Daily Level Forecasting}
The daily test set contains 365 days. The weather-only previous-day ridge model reaches $R^2=0.3600$ and RMSE=18.22. Enhanced models improve performance. The validation-weighted ensemble achieves the best daily RMSE, 16.97, and $R^2=0.4447$. Compact enhanced ridge has nearly the same $R^2=0.4442$. Both outperform the persistence baseline, whose $R^2$ is only 0.0738.

\begin{table}[!htbp]
\centering
\caption{Daily PM$_{2.5}$ Forecasting Performance}
\small
\begin{tabular}{lrrrr}
\toprule
Model & RMSE & MAE & Bias & $R^2$ \\
\midrule
Weather-only previous-day ridge & 18.22 & 13.32 & 4.07 & 0.3600 \\
Compact enhanced ridge & 16.98 & 11.82 & 0.16 & 0.4442 \\
All-lag enhanced ridge & 17.32 & 12.00 & -1.05 & 0.4216 \\
Enhanced lag random forest & 17.54 & 11.67 & 0.02 & 0.4065 \\
Historical analog KNN & 18.77 & 13.34 & 0.11 & 0.3205 \\
Validation-weighted ensemble & 16.97 & 11.66 & 0.14 & 0.4447 \\
Persistence baseline & 21.91 & 15.10 & 0.09 & 0.0738 \\
\bottomrule
\end{tabular}
\end{table}

In classification terms, the validation-weighted ensemble has 60.82\% three-class accuracy and weighted F1 of 0.604. Its high-pollution precision is 0.648 and high-pollution recall is 0.793. This means the ensemble is useful for identifying many high-pollution days, but there is still room to reduce false alarms and improve exact concentration accuracy.

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.88\textwidth]{best_ensemble_observed_vs_predicted.png}
\caption{Best ensemble daily PM$_{2.5}$ observed-versus-predicted plot.}
\end{figure}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.95\textwidth]{daily_prediction_timeseries.png}
\caption{Daily PM$_{2.5}$ prediction time series for the test period.}
\end{figure}

\subsection{Multi-Horizon Trend Forecasting}
The trend experiment predicts 1--7 day horizons. The direct ridge model is strongest for horizon 1 with RMSE=18.17, $R^2=0.3701$, and direction accuracy 0.738. For horizons 2--7, $R^2$ falls to roughly 0.08--0.16, while direction accuracy remains around 0.71--0.79 for the better models. This means short-term direction is more predictable than exact concentration at longer horizons.

\begin{table}[!htbp]
\centering
\caption{Selected Multi-Horizon Forecasting Results}
\small
\begin{tabular}{lrrrr}
\toprule
Model and horizon & RMSE & MAE & $R^2$ & Direction accuracy \\
\midrule
Direct ridge H1 & 18.17 & 12.43 & 0.3701 & 0.738 \\
Optimized ensemble H1 & 18.53 & 12.86 & 0.3451 & 0.719 \\
Analog KNN H2 & 20.97 & 14.96 & 0.1589 & 0.728 \\
Analog KNN H3 & 21.04 & 14.86 & 0.1510 & 0.789 \\
Analog KNN H5 & 21.06 & 14.86 & 0.1483 & 0.752 \\
Analog KNN H7 & 21.31 & 15.19 & 0.1245 & 0.712 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[!htbp]
\centering
\includegraphics[width=0.88\textwidth]{trend_horizon_rmse.png}
\caption{Forecast RMSE by horizon.}
\end{figure}
\FloatBarrier

\section{Integrated Discussion}
The complete evidence chain is internally consistent. The Bayesian network shows that high CO, high NO$_2$, high PM$_{10}$, low wind speed, and high humidity raise the probability of high PM$_{2.5}$ under concrete thresholds. Random forest and LASSO then show that CO, NO$_2$, and lagged PM$_{2.5}$ remain important in predictive and regularized settings. Full OLS gives interpretable conditional coefficients and confirms that adding gaseous pollutants is essential. GAM shows that several relationships, especially CO, NO$_2$, humidity, wind speed, and O$_3$, contain nonlinear patterns.

The most robust conclusion is not that one weather variable alone determines PM$_{2.5}$. Instead, PM$_{2.5}$ is shaped by three layers:
\begin{enumerate}
\item Core pollutant and persistence layer: CO, NO$_2$, and lagged PM$_{2.5}$ are stable across methods.
\item Meteorological state layer: low wind speed, high humidity, and no-rain states alter high-pollution risk, often through threshold-like patterns.
\item Nonlinear correction layer: GAM shows that the effect of wind, humidity, O$_3$, CO, and NO$_2$ changes across their observed ranges.
\end{enumerate}

The forecasting section supports, but does not replace, the mechanism analysis. Forecasting models show that previous-day and lagged information can predict PM$_{2.5}$ better than persistence alone, but the daily $R^2$ remains around 0.445 for the best ensemble. This is lower than the full contemporaneous OLS/GAM mechanism models because the forecasting task uses information available before the target day and must predict future conditions.

\section{Conclusion}
This revised report resolves the Step 5 inconsistency and integrates the prediction work into a separate, coherent extension. The formal mechanism model is the full OLS with meteorology, gaseous pollutants, lagged PM$_{2.5}$, and seasonal controls. Its $R^2=0.7698$ confirms that gaseous pollutants are indispensable for explaining PM$_{2.5}$ variation. The weather-only OLS, with $R^2=0.4744$, is useful only as a forecasting-branch baseline.

Detailed threshold results make the conclusions more concrete. Low wind speed ($\leq1.65$ m/s) raises high-pollution probability to 41.55\%, while high wind speed ($>2.40$ m/s) lowers it to 24.29\%. High CO ($>0.54$ mg/m$^3$) raises high-pollution probability to 75.36\%, and high NO$_2$ ($>25.92$ $\mu$g/m$^3$) raises it to 64.77\%. GAM confirms that CO is significantly positive from 0.54--1.08 mg/m$^3$ and NO$_2$ from 25.43--58.67 $\mu$g/m$^3$. Therefore, the final interpretation should emphasize concrete ranges, cross-method agreement, and nonlinear atmospheric mechanisms rather than broad variable labels.

\begin{thebibliography}{9}
\bibitem{breiman2001} Breiman, L. (2001). Random forests. \emph{Machine Learning}, 45(1), 5--32.
\bibitem{eilers1996} Eilers, P. H. C., and Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. \emph{Statistical Science}, 11(2), 89--121.
\bibitem{hastie1990} Hastie, T. J., and Tibshirani, R. J. (1990). \emph{Generalized Additive Models}. Chapman and Hall.
\bibitem{koller2009} Koller, D., and Friedman, N. (2009). \emph{Probabilistic Graphical Models}. MIT Press.
\bibitem{kutner2005} Kutner, M. H., Nachtsheim, C. J., Neter, J., and Li, W. (2005). \emph{Applied Linear Statistical Models}. McGraw-Hill/Irwin.
\bibitem{pearl1988} Pearl, J. (1988). \emph{Probabilistic Reasoning in Intelligent Systems}. Morgan Kaufmann.
\bibitem{tibshirani1996} Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. \emph{Journal of the Royal Statistical Society: Series B}, 58(1), 267--288.
\bibitem{wood2017} Wood, S. N. (2017). \emph{Generalized Additive Models: An Introduction with R}. Chapman and Hall/CRC.
\bibitem{wooldridge2016} Wooldridge, J. M. (2016). \emph{Introductory Econometrics: A Modern Approach}. Cengage Learning.
\end{thebibliography}

\end{document}
"""
    TEX_PATH.write_text(tex.strip() + "\n", encoding="utf-8")


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(A4[0] / 2, 0.35 * inch, f"{doc.page}")
    canvas.restoreState()


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(data, widths=None, font_size=8.0):
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def figure(name: str, caption: str, max_width: float = 6.8 * inch):
    path = ASSET_DIR / name
    if not path.exists():
        return []
    with PILImage.open(path) as img:
        w, h = img.size
    ratio = h / w
    width = max_width
    height = width * ratio
    if height > 4.6 * inch:
        height = 4.6 * inch
        width = height / ratio
    return [
        Spacer(1, 0.08 * inch),
        Image(str(path), width=width, height=height, hAlign="CENTER"),
        Paragraph(f"<i>{caption}</i>", caption_style),
        Spacer(1, 0.12 * inch),
    ]


styles = getSampleStyleSheet()
title_style = ParagraphStyle(
    "TitleCustom",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=18,
    leading=22,
    alignment=TA_CENTER,
    spaceAfter=12,
)
h1 = ParagraphStyle("Heading1Custom", parent=styles["Heading1"], fontSize=14, leading=18, spaceBefore=12, spaceAfter=6)
h2 = ParagraphStyle("Heading2Custom", parent=styles["Heading2"], fontSize=11.5, leading=15, spaceBefore=8, spaceAfter=4)
body = ParagraphStyle("BodyCustom", parent=styles["BodyText"], fontSize=9.2, leading=12.2, alignment=TA_JUSTIFY, spaceAfter=5)
small = ParagraphStyle("SmallCustom", parent=body, fontSize=8.2, leading=10.5, spaceAfter=4)
caption_style = ParagraphStyle("CaptionCustom", parent=small, alignment=TA_CENTER, textColor=colors.HexColor("#475569"))


def write_pdf() -> None:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.55 * inch,
    )
    story = []
    story.append(Paragraph("Integrated PM2.5 Analysis and Forecasting Report", title_style))
    story.append(Paragraph("Beijing Daily Air Pollution: Mechanism, Nonlinearity, and Prediction", caption_style))
    story.append(Paragraph(date.today().isoformat(), caption_style))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("<b>Abstract.</b> This report rewrites the PM2.5 analysis into three connected modules: problem discovery, key-variable identification, and mechanism explanation. Module 1 combines descriptive statistics and Bayesian-network risk analysis. Module 2 combines random forest and LASSO. Module 3 combines the unified full OLS model and GAM nonlinear diagnostics. A separate forecasting extension evaluates lagged-feature prediction. The key correction is that Step 5 uses the full OLS model with meteorology, gaseous pollutants, lagged PM2.5, and seasonal controls. The full OLS reaches R2=0.7698, while the weather-only OLS reaches only R2=0.4744 and is treated as a forecasting-branch baseline.", body))

    sections = [
        ("1. Research Question and Unified Workflow", [
            "The main question is how meteorological conditions, gaseous pollutants, and short-term pollution persistence jointly influence daily PM2.5 in Beijing, and whether lagged information can support short-term forecasting.",
            "The mechanism part is now organized into three modules. Module 1 discovers the problem through descriptive statistics and Bayesian-network risk states. Module 2 identifies key variables through random forest and LASSO. Module 3 explains influence mechanisms through full OLS and GAM. Forecasting remains a separate extension because it uses previous-day and historical information rather than the same contemporaneous information set.",
        ]),
        ("2. Data and Variable Construction", [
            "The daily dataset contains 1,462 observations. After constructing lagged PM2.5, the main regression sample contains 1,461 valid observations. Predictors include temperature, humidity, wind speed, precipitation, CO, NO2, SO2, O3, lagged PM2.5, month, and seasonal indicators.",
            "High PM2.5 is defined as PM2.5 > 32.24 ug/m3. Key BN cutpoints include low wind speed <=1.65 m/s, high wind speed >2.40 m/s, high humidity >67.29%, high CO >0.54 mg/m3, and high NO2 >25.92 ug/m3.",
        ]),
    ]
    for heading, paras in sections:
        story.append(Paragraph(heading, h1))
        for p in paras:
            story.append(Paragraph(p, body))

    story.append(Paragraph("3. Module 1: Discovering the Problem", h1))
    story.append(Paragraph("This module combines Step 1 descriptive statistics and Step 2 Bayesian-network risk analysis. It answers what distributional features PM2.5 has and under which concrete states high pollution is more likely.", body))
    story.append(Paragraph("Step 1: Descriptive Statistics", h2))
    story.append(Paragraph("PM2.5 is right-skewed: mean 30.10 ug/m3, median 22.71, and maximum 164.34. Lagged PM2.5 has almost the same distribution as current PM2.5, showing short-term persistence. Precipitation has median zero and a high maximum, so it should be interpreted both as rain/no rain and as a continuous amount.", body))
    story.append(table([
        ["Variable", "n", "Mean", "SD", "Median", "Max"],
        ["PM2.5", "1462", "30.10", "25.72", "22.71", "164.34"],
        ["Wind speed", "1462", "2.23", "1.02", "2.01", "9.71"],
        ["CO", "1462", "0.49", "0.20", "0.46", "1.45"],
        ["NO2", "1462", "23.14", "12.43", "20.10", "66.21"],
        ["Lag PM2.5", "1461", "30.11", "25.72", "22.75", "164.34"],
    ], [1.5 * inch, 0.6 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch]))
    story.extend(figure("desc_summary_panels.png", "Figure 1. Descriptive time series, histogram, boxplots, and correlation heatmap."))

    story.append(Paragraph("Step 2: Bayesian Network and Concrete Risk States", h2))
    story.append(Paragraph("The Bayesian network converts variable states into high-pollution probabilities. The baseline high-pollution probability is 33.38%. Concrete state definitions are the core contribution of this step.", body))
    story.append(table([
        ["State", "Concrete definition", "BN P(high)", "Change", "Mean PM2.5"],
        ["Low wind", "<=1.65 m/s", "41.55%", "+8.17 pp", "36.88"],
        ["High wind", ">2.40 m/s", "24.29%", "-9.09 pp", "23.70"],
        ["High CO", ">0.54 mg/m3", "75.36%", "+41.98 pp", "52.95"],
        ["High NO2", ">25.92 ug/m3", "64.77%", "+31.39 pp", "49.45"],
        ["High humidity", ">67.29%", "40.29%", "+6.91 pp", "36.35"],
        ["High PM10", ">62.30 ug/m3", "84.11%", "+50.74 pp", "55.95"],
    ], [1.05 * inch, 1.75 * inch, 0.85 * inch, 0.75 * inch, 0.85 * inch]))
    story.append(Paragraph("The correct wind-speed conclusion is specific: low wind speed is a higher-risk stagnation state and high wind speed is a lower-risk dispersion state. This is more precise than simply saying that wind speed affects PM2.5.", body))
    story.extend(figure("bn_summary_panels.png", "Figure 2. Bayesian-network structure, risk states, sensitivity ranking, and key scenarios."))

    story.append(PageBreak())
    story.append(Paragraph("4. Module 2: Identifying Key Variables", h1))
    story.append(Paragraph("This module combines Step 3 random forest and Step 4 LASSO. It answers which variables repeatedly appear important in nonlinear prediction and regularized screening under collinearity.", body))
    story.append(Paragraph("Step 3: Random Forest Regression", h2))
    story.append(Paragraph("Random forest provides a nonlinear predictive benchmark. It reaches test RMSE=11.64, MAE=7.13, test R2=0.7787, and OOB R2=0.7876. CO accounts for 57.77% of feature importance, lagged PM2.5 for 18.80%, and NO2 for 8.46%.", body))
    story.extend(figure("rf_summary_panels.jpg", "Figure 3. Random-forest feature importance and diagnostics."))

    story.append(Paragraph("Step 4: LASSO Variable Selection", h2))
    story.append(Paragraph("LASSO screens variables under regularization and collinearity. Cross-validation selects alpha_min=0.0517. The largest positive standardized coefficients are CO=14.300, NO2=7.946, lagged PM2.5=6.952, Spring=2.455, and O3=1.734. This agrees with the BN and random-forest evidence.", body))
    story.extend(figure("lasso_summary_panels.png", "Figure 4. LASSO cross-validation, coefficient path, and selected coefficients."))

    story.append(Paragraph("5. Module 3: Explaining the Influence Mechanism", h1))
    story.append(Paragraph("This module combines Step 5 full OLS and Step 6 GAM. It answers the direction, significance, and nonlinear shape of the key effects.", body))
    story.append(Paragraph("Step 5: Unified Full OLS Model", h2))
    story.append(Paragraph("The formal Step 5 mechanism model is the full OLS with meteorology, CO, NO2, SO2, O3, lagged PM2.5, and season. This model must replace the old weather-only OLS in the main report. The full OLS has R2=0.7698 and adjusted R2=0.7679. The weather-only OLS has R2=0.4744 and should be used only as a forecasting-branch baseline.", body))
    story.append(table([
        ["Model", "R2", "Adj. R2", "RMSE", "Use"],
        ["Full OLS", "0.7698", "0.7679", "12.34", "Mechanism Step 5"],
        ["Weather-only OLS", "0.4744", "0.4715", "18.64", "Forecast branch only"],
    ], [2.2 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch, 1.55 * inch]))
    story.append(table([
        ["Variable", "Raw coef.", "Std. coef.", "p", "Meaning"],
        ["CO", "71.96", "0.565", "<0.001", "strongest positive association"],
        ["NO2", "0.635", "0.307", "<0.001", "stable gaseous-pollutant term"],
        ["Lag PM2.5", "0.271", "0.271", "<0.001", "short-term persistence"],
        ["Wind speed", "1.047", "0.042", "0.021", "small conditional linear term"],
        ["Temperature", "-0.147", "-0.068", "0.041", "weak negative after controls"],
    ], [1.1 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch, 2.0 * inch]))
    story.append(Paragraph("The wind coefficient is small and conditional. It does not overturn the BN finding that low wind speed is a higher-risk state; instead, it shows that wind speed has a nonlinear and collinearity-sensitive relationship once pollutants and season are controlled.", body))
    story.extend(figure("mlr_diagnostic_panels.png", "Figure 5. Full OLS diagnostics."))

    story.append(Paragraph("Step 6: GAM Nonlinear Diagnostics", h2))
    story.append(Paragraph("GAM reaches R2=0.7930, adjusted R2=0.7881, RMSE=11.70, and an AIC that is 183.89 lower than the linear baseline. The smooth terms show where the effects become strong.", body))
    story.append(table([
        ["Smooth term", "Drop dev.", "Concrete nonlinear interpretation"],
        ["CO", "0.0692", "positive and significant from 0.54-1.08 mg/m3"],
        ["NO2", "0.0544", "positive and significant from 25.43-58.67 ug/m3"],
        ["O3", "0.0187", "negative at 8.65-37.46, positive at 46.22-155.20"],
        ["Humidity", "0.0072", "positive at 73.82-91.28%, negative at 52.29-66.84%"],
        ["Wind speed", "0.0020", "negative around 1.79-1.94 and 2.25-2.37 m/s"],
    ], [1.25 * inch, 0.8 * inch, 4.3 * inch], font_size=7.6))
    story.extend(figure("gam_diagnostic_importance_panels_clean.png", "Figure 6. GAM diagnostics and smooth-term ranking."))

    story.append(PageBreak())
    story.append(Paragraph("6. Forecasting Extension: Predicting PM2.5", h1))
    story.append(Paragraph("The forecasting experiment is a separate extension. It asks whether previous-day and historical features can predict future PM2.5. It should not replace the full OLS mechanism model because it uses a different information set.", body))
    story.append(table([
        ["Daily model", "RMSE", "MAE", "Bias", "R2"],
        ["Weather-only previous-day ridge", "18.22", "13.32", "4.07", "0.3600"],
        ["Compact enhanced ridge", "16.98", "11.82", "0.16", "0.4442"],
        ["All-lag enhanced ridge", "17.32", "12.00", "-1.05", "0.4216"],
        ["Enhanced lag random forest", "17.54", "11.67", "0.02", "0.4065"],
        ["Validation-weighted ensemble", "16.97", "11.66", "0.14", "0.4447"],
        ["Persistence baseline", "21.91", "15.10", "0.09", "0.0738"],
    ], [2.45 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch]))
    story.append(Paragraph("The validation-weighted ensemble has the best daily RMSE and R2. For three-class level prediction, it reaches 60.82% accuracy and weighted F1=0.604, with high-pollution precision=0.648 and high-pollution recall=0.793.", body))
    story.extend(figure("best_ensemble_observed_vs_predicted.png", "Figure 7. Best ensemble observed-versus-predicted daily PM2.5."))
    story.extend(figure("daily_prediction_timeseries.png", "Figure 8. Daily prediction time series."))
    story.append(Paragraph("For multi-horizon trend forecasting, direct ridge performs best at H1 with R2=0.3701 and direction accuracy=0.738. At horizons 2-7, R2 falls to about 0.08-0.16, while direction accuracy remains around 0.71-0.79 for better models. This means short-term direction is more predictable than exact concentration at longer horizons.", body))
    story.extend(figure("trend_horizon_rmse.png", "Figure 9. Forecast RMSE by horizon."))

    story.append(Paragraph("7. Integrated Conclusion", h1))
    conclusions = [
        "CO, NO2, and lagged PM2.5 are the most stable variables across BN, RF, LASSO, OLS, and GAM.",
        "Low wind speed <=1.65 m/s raises high-pollution probability to 41.55%, while high wind speed >2.40 m/s lowers it to 24.29%.",
        "High CO >0.54 mg/m3 raises high-pollution probability to 75.36%, and high NO2 >25.92 ug/m3 raises it to 64.77%.",
        "The full OLS must be the formal Step 5 model because its R2=0.7698, compared with 0.4744 for weather-only OLS.",
        "GAM confirms nonlinear ranges, especially for CO, NO2, humidity, wind speed, and O3.",
        "Forecasting models improve over persistence but remain harder than contemporaneous mechanism modeling; the best daily ensemble has R2=0.4447.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(c, body)) for c in conclusions], bulletType="bullet", leftIndent=18))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>References.</b> Breiman (2001); Eilers and Marx (1996); Hastie and Tibshirani (1990); Koller and Friedman (2009); Kutner et al. (2005); Pearl (1988); Tibshirani (1996); Wood (2017); Wooldridge (2016).", small))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def write_build_notes() -> None:
    note = """# Build Notes

This directory contains:

- `PM25_integrated_full_report.tex`: the complete LaTeX source.
- `PM25_integrated_full_report.pdf`: a PDF rendering of the same report content.
- `report_assets/`: figures required by the TeX file.

The current runtime did not have `pdflatex`, `xelatex`, `lualatex`, `bibtex`, or `latexmk` installed. A portable Tectonic compiler download was attempted but was too slow to finish in the active session, so the PDF was rendered from the same report content with the bundled Python/reportlab stack.

To compile the TeX source in a LaTeX environment, run one of:

```bash
pdflatex PM25_integrated_full_report.tex
pdflatex PM25_integrated_full_report.tex
```

or:

```bash
tectonic PM25_integrated_full_report.tex
```

The TeX file is self-contained and uses `thebibliography`, so no BibTeX run is required.
"""
    (REPORT_DIR / "BUILD_NOTES.md").write_text(note, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    copy_assets()
    write_tex()
    write_pdf()
    write_build_notes()
    print(TEX_PATH)
    print(PDF_PATH)


if __name__ == "__main__":
    main()
