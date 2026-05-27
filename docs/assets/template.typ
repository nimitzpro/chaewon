#let sig_style(body) = {
  set text(font: "Apple Chancery", size: 12pt)
  body
}

#let project(
  title: "",
  author: "",
  supervisors: (),
  date: "",
  degree: "",
  university: "",
  signature: "",
  sig_date: "",
  logo: none,
  abstract: none,
  body,
) = {
  // Document and page setup matching the LaTeX a4paper, 12pt
  set page(paper: "a4")
  set text(font: "Libertinus Serif", size: 12pt)
  // set text(font: "Helvetica", size: 12pt)

  // Configure link formatting and equation numbering
  show link: underline
  set math.equation(numbering: "(1)")

  // Title Page Setup
  align(center)[
    #v(12pt)
    #text(size: 18pt, weight: "bold", title)

    #v(6pt)
    #text(size: 14pt, style: "italic", author)

    #v(24pt)
    #if logo != none [
      #logo
    ] else [
      #v(100pt) // Vertical spacing if no logo is provided
    ]

    #v(24pt)
    #for supervisor in supervisors [
      #text(style: "italic", supervisor) \
    ]

    #v(6pt)
    #text(weight: "bold", date) \
    #text(weight: "bold", degree) \
    #text(weight: "bold", university)
  ]

  pagebreak()

  // Abstract Section
  if abstract != none [
    #heading(level: 1, numbering: none)[Abstract]
    #abstract
    #pagebreak()
  ]

  // Declaration of Originality
  heading(level: 1, numbering: none)[Declaration of Originality]
  [
    In signing this declaration, you are confirming, in writing, that the submitted work is entirely your own original work, except where clearly attributed otherwise, and that it has not been submitted partly or wholly for any other educational award.

    I hereby declare that:
    - this is all my own work, unless clearly indicated otherwise, with full and proper accreditation;
    - with respect to my own work: none of it has been submitted at any educational institution contributing in any way to an educational award;
    - with respect to another's work: all text, diagrams, code, or ideas, whether verbatim, paraphrased or otherwise modified or adapted, have been duly attributed to the source in a scholarly manner.

    #v(1cm)
    *Signed:* #box(signature, width: 120pt, height: 12pt, stroke: (bottom: 1pt)) \ \
    *Date:* #box(sig_style(sig_date), width: 133pt, height: 12pt, stroke: (bottom: 1pt))
  ]

  pagebreak()

  // Table of Contents
  outline(title: "Contents", indent: auto)

  pagebreak()

  // Main Body Typography Settings
  set heading(numbering: none)
  set par(justify: true, leading: 0.65em)

  body
}
