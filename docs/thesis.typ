#import "./assets/template.typ": project

#import "@preview/lilaq:0.5.0" as lq
#import "@preview/intextual:0.1.1": *

// intertext-rule show rule is required for elements to display properly.
#show: intertext-rule
#show math.equation: set text(size: 12pt)

// #set text(font: "Noto Sans CJK JP", size: 12pt)
// #let mono(body) = {
//   set text(font: "Noto Sans Mono CJK JP")
//   body
// }

#show: project.with(
  title: [Comparing Volatility Models using a \ Sig-MMD Diagnostic],
  author: "Alexander Stradnic",
  supervisors: ("Professors Cónall Kelly and Mingchuan Zhao",),
  date: "August 2026",
  degree: "MSc. Financial and Computational Mathematics",
  university: "University College Cork",
  logo: image("./assets/logo.png", width: 60%),
  abstract: [
    This thesis explores the application of the signature maximum mean discrepancy (sig-MMD) diagnostic to evaluate and compare various stochastic volatility models. We examine the capacity of sig-MMD to distinguish between classical models and rough volatility paradigms using both synthetic paths and historical market data. By tracking pairwise model distances over multiple market regimes, this research provides a novel path-distributional approach to benchmark stochastic volatility against real market paths.
  ],
  signature: [#emph[Alexander Stradnic]],
  sig_date: [X#super[th] August 2026],
)

= Introduction
#cite(<sig_book>)
#cite(<sigMMD>)
#cite(<sigker_goursat>)

= Models

= Calibration & Diagnostic Methods

= Synthetic Validation

= Real Market Calibration (static)

= Models vs Real Market Dynamics: Real-Path Diagnostic

= Dynamic Experiments

= Conclusions

= Data
Oxford-Man Institute’s Realized Volatility Indices: #link("https://github.com/jonathancornelissen/highfrequency")

#bibliography("./citations.bib", style: "ieee")