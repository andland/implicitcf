#' Collaborative Filtering for Implicit Feedback Datasets
#'
#' @param R A sparse implicit feedback matrix, where the rows typically represent users
#'   and the columns typically represent items. The elements of the matrix represent the
#'   number of times that the users have interacted with the items
#' @param alpha Used to calculate cost matrix \code{C} = 1 + \code{alpha} * \code{R}
#'   if \code{C1} is not specified
#' @param C1 Equal the cost matrix (\code{C}) minus 1, which should be sparse
#' @param P A binary matrix, indicating whether or not the users interacted with the items
#' @param f The rank of the matrix factorization
#' @param lambda The L2 squared norm penalty on the latent row and column features
#' @param init_stdv Standard deviation to initialize the latent row and column features
#' @param max_iters How many iterations to run the algorithm for
#' @param parallel Whether to use \code{foreach} package to parallelize the computation.
#'    See the example for how to use. Does not work for Windows.
#' @param quiet Whether or not to print out progress
#'
#' @return An S3 object of class \code{implicitcf} which is a list with the following components:
#'   \item{X}{the rank-\code{f} latent features for the users}
#'   \item{Y}{the rank-\code{f} latent features for the items}
#'   \item{loss_trace}{the loss function after each iteration. It should be non-increasing}
#'   \item{f}{the rank used}
#'   \item{lambda}{the penalty parameter used}
#'
#' @details This function impliments the algorithm of Hu et al. (2008) in R using sparse matrices.
#'   It solves for \code{X} and \code{Y} by minimizing the loss function:
#'   \deqn{\sum_{u, i} c_{ui} (p_{ui} - x_u^Ty_i)^2 + \lambda (||X||_F^2 + ||Y||_F^2)}
#'
#'   It does this by iteratively solving for \eqn{x_u, u = 1, ...,}\code{nrow(R)} and
#'   \eqn{y_i, i = 1, ...,}\code{ncol(R)}, holding everything else constant.
#'
#'   Since implicit feedback data is typically sparse, the algorithm and this code are optimized
#'   take advantage of the sparsity. That being said, the algorithm involves looping over
#'   the rows and columns of the matrix, which R is slow at.
#'
#' @references
#' Hu, Y., Koren, Y., Volinsky, C., 2008. Collaborative filtering for implicit feedback datasets.
#' In Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on (pp. 263-272). IEEE.
#'
#' @examples
#'  X = matrix(rnorm(10 * 2, 0, 1), 10, 2)
#'  Y = matrix(rnorm(5 * 2, 0, 2), 5, 2)
#'  noise = matrix(rnorm(10 * 5, 0, 0.5), 10, 5)
#'  R = round(pmax(tcrossprod(X, Y) + noise, 0))
#'
#'  icf = implicitcf(R, f = 2, alpha = 1, lambda = 0.1, quiet = FALSE)
#'
#'  # should be decreasing
#'  plot(icf$loss_trace)
#'
#'  # to use parallel
#'  \dontrun{
#'  require(doMC)
#'  registerDoMC(cores = parallel::detectCores())
#'  icf = implicitcf(R, f = 2, alpha = 1, lambda = 0.1, quiet = FALSE, parallel = TRUE)
#'  }
#'
#' @export
implicitcf <- function(R, alpha = 1, C1 = alpha * R, P = (R > 0) * 1,
                       f = 10, lambda = 0,
                       init_stdv = ifelse(lambda == 0, 0.01, 1 / sqrt(2 * lambda)),
                       max_iters = 10, quiet = TRUE, parallel = FALSE) {
  # check R, C1, and P dimensions and 0's match up
  stopifnot(all(dim(C1) == dim(P)))
  if (!all(which(C1 > 0) == which(P > 0))) {
    warning("non-zero elements of C1 and P do not match. This could cause issues in the algorithm")
  }
  # C1, and P are sparseMatrix class
  if (!inherits(C1, "sparseMatrix")) {
    C1 = Matrix::Matrix(C1, sparse = TRUE)
  }
  if (!inherits(P, "sparseMatrix")) {
    P = Matrix::Matrix(P, sparse = TRUE)
  }
  # R doesn't need to be specified as long as C1 and P are
  if (!missing(R)) {
    rm(R)
  }

  if (parallel) {
    if (!requireNamespace("foreach", quietly = TRUE)) {
      warning("foreach package not installed. Setting parallel = FALSE.\n",
              "Install foreach package to use parallel.")
      parallel = FALSE
    }
  }

  nrows = nrow(P)
  ncols = ncol(P)

  # f is typically small, so I don't know if this helps
  Lambda = Matrix::Diagonal(f, x = lambda)
  # Lambda = diag(x = lambda, f, f)
  CP = C1 * P + P

  # only initialize X so that it knows the size
  X = Matrix::Matrix(rnorm(nrows * f, 0, init_stdv), nrows, f)
  Y = Matrix::Matrix(rnorm(ncols * f, 0, init_stdv), ncols, f)

  loss_trace = rep(NA, max_iters)
  for (iter in 1:max_iters) {
    if (!quiet) {
      cat("Iter", iter, "-- ")
    }

    YtY = Matrix::crossprod(Y)
    for (u in 1:nrows) {
      # TODO: compare diag(C1[u, ]) to Diagonal(x = C1[u, ]) since Diagonal keeps 0's
      inv = YtY + Matrix::t(Y) %*% diag(C1[u, ]) %*% Y + Lambda

      # TODO: compare to rhs = t(Y) %*% CP[u, ], to make sure it gives same results
      # rhs = t(Y) %*% diag(C1[u, ] + 1) %*% P[u, ]
      rhs = Matrix::t(Y) %*% CP[u, ]

      x = Matrix::solve(inv, rhs)
      X[u, ] = as.numeric(x)
    }

    XtX = Matrix::crossprod(X)
    for (i in 1:ncols) {
      # TODO: compare diag(C1[, i]) to Diagonal(x = C1[, i]) since Diagonal keeps 0's
      inv = XtX + Matrix::t(X) %*% diag(C1[, i]) %*% X + Lambda

      # TODO: compare to rhs = t(X) %*% CP[, i], to make sure it gives same results
      # rhs = t(X) %*% diag(C1[, i] + 1) %*% P[, i]
      rhs = Matrix::t(X) %*% CP[, i]

      y = Matrix::solve(inv, rhs)
      Y[i, ] = as.numeric(y)
    }

    loss_trace[iter] = sum((C1 + 1) * (P - Matrix::tcrossprod(X, Y))^2) + lambda * (sum(X^2) + sum(Y^2))
    if (!quiet) {
      cat("Loss =", loss_trace[iter], "\n")
    }
  }
  structure(
    list(
      X = X,
      Y = Y,
      loss_trace = loss_trace,
      # alpha = alpha,
      f = f,
      lambda = lambda
    ),
    class = "implicitcf"
  )
}
