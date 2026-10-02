# FLSSS probe driver (task 1.11). Called once per instance solve by
# src/solvers/flsss_solver.py.
#
# stdin (plain text): line 1 = "TL TM TR eps tlimit"; line 2 = items
# (space-separated).
# stdout: one line per feasible alpha-probe: "SUBSET <0-based original ids>"
#         (subset size is freed by padding with n zero-items; pads dropped).
#
# The binary search on alpha mirrors the certified alpha-cut FPTAS
# (probe at alpha=1, ceil(log2(1/eps)) bisection steps, fallback at
# alpha=eps); every feasible probe's subset is emitted so the Python side
# scores each through the shared protocol objective.

suppressMessages(library(FLSSS))

con <- file("stdin")
lines <- readLines(con)
close(con)
hdr <- as.numeric(strsplit(trimws(lines[1]), " +")[[1]])
TL <- hdr[1]; TM <- hdr[2]; TR <- hdr[3]; eps <- hdr[4]; tlimit <- hdr[5]
items <- as.numeric(strsplit(trimws(lines[2]), " +")[[1]])
n <- length(items)

padded <- c(items, rep(0, n))
ord <- order(padded)
sorted_v <- padded[ord]

probe <- function(alpha) {
  L <- TL + alpha * (TM - TL)
  R <- TR - alpha * (TR - TM)
  # Items are integers, so the attainable sums in [L, R] are exactly the
  # integers in [ceil(L), floor(R)]. Probing that integer window with a
  # half-unit margin admits the same sums but never asks FLSSS for ME = 0,
  # for which it returns no solution (the exact-hit probe at alpha = 1
  # silently failed before this fix, 2026-10-02).
  L <- ceiling(L); R <- floor(R)
  if (R < L) return(FALSE)
  target <- (L + R) / 2
  ME <- (R - L) / 2 + 0.5
  res <- tryCatch(
    FLSSS(len = n, v = sorted_v, target = target, ME = ME,
          solutionNeed = 1, tlimit = tlimit),
    error = function(e) list())
  if (length(res) == 0) return(FALSE)
  sel_sorted <- res[[1]]
  orig <- ord[sel_sorted]           # positions in the padded vector
  real <- orig[orig <= n]           # drop zero pads
  cat("SUBSET", paste(real - 1, collapse = " "), "\n")
  TRUE
}

invisible(probe(1.0))
lo <- 0.0; hi <- 1.0
for (i in seq_len(max(1, ceiling(log2(1 / eps))))) {
  mid <- 0.5 * (lo + hi)
  if (probe(mid)) lo <- mid else hi <- mid
}
if (lo == 0.0) invisible(probe(eps))
