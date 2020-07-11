#' scMetric: metric learning and visualization for scRNA-seq data
#'
#' Apply a weakly supervised metric learning algorithm ITML to scRNA-seq data.
#' Users give very few training samples to tell expected angle they would use
#' to analyze the data, and the function learns the metric automatically for
#' downstream clustering and visualization.
#'
#' @param X a scRNA-seq expression matrix, cells for rows and genes for columns.
#' @param label a vector. Specify which group cells belong to, corresponding to rows in X. If NULL(default), \code{constraints} should not be NULL.
#' @param constraints a N by 3 matrix, weak supervision information. N stands for total number of cell pairs. The first 2 columns specify two cells. The 3rd column is a value specifying whether corresponding two cells in the first two columns are similar, 1 for similar and -1 for dissimilar. If NULL(default), \code{label} cannot be NULL and \code{num_constraints} pairs will be chosen randomly according to \code{label} for metric learning. Cells that have the same label are similar. Otherwise, they are dissimilar.
#' @param num_constraints total number of similar and dissimilar pairs that are used. No larger than N. If \code{constraints} is not NULL, first \code{num_constraints} rows of \code{constraints} will be used. Default: 100
#' @param thresh threshold that decides when metric learning iteration stops. Default: 0.01
#' @param max_iters max iterations of metric learning. Default: 100000
#' @param draw_tSNE boolean. Default: FALSE. Specify whether to draw tSNE plot or not
#'
#' @return
#' List containing four outputs:
#' \itemize{
#' \item newData: new data based on new metric, rows are cells and columns are linear combination of original genes expressions
#' \item newMetric: learned metric, a d by d matric where d represents genes numbers
#' \item constraints: constraints used for metric learning
#' \item sortGenes: genes sorted by importance score
#' }
#'
#' @examples
#' data(testData)
#' res <- scMetric(counts, label = label1, num_constraints = 50, thresh = 0.1, draw_tSNE = TRUE)
#'
#' @importFrom Rtsne Rtsne
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 guides rel element_text element_line xlab ylab aes geom_point scale_color_brewer theme_bw theme guide_legend
#' @export


scMetric <- function(X, label = NULL, constraints = NULL, num_constraints = 100, thresh = 10e-3, max_iters = 100000, draw_tSNE = FALSE){

  # Invalid input control
  if(!is.matrix(X) & !is.data.frame(X))
    stop("Wrong data type of 'X'")
  if(sum(is.na(X)) > 0)
    stop("NA detected in 'X'");gc();
  if(sum(X < 0) > 0)
    stop("Negative value detected in 'X'");gc();
  if(all(X == 0))
    stop("All elements of 'X' are zero");gc();
  if(any(colSums(X) == 0))
    warning("Library size of zero detected in 'X'");gc();

  if(!is.null(label)){
    if(nrow(X) != length(label))
      stop("Row number of 'label' must equal to row number of 'X'")
  }
  if(!is.null(constraints)){
    if(!is.matrix(constraints) & !is.data.frame(constraints))
      stop("Wrong data type of 'constraints'")
    if(ncol(constraints) != 3)
      stop("Wrong format of 'constraints'")
    if((sum(constraints[,3] == 1) + sum(constraints[,3] == -1)) != nrow(constraints))
      stop("Wrong value in 3rd colume of 'constraints'. Must be 1 for similar pairs and -1 for dissimilar pairs")
    if(num_constraints > nrow(constraints))
      stop("No enough constraints!Set 'num_constraints' smaller!")
  }
  if(!is.numeric(num_constraints))
    stop("Wrong data type of 'num_constraints'")
  if(round(num_constraints) != num_constraints)
    stop("'num_constraints' should be integer")
  if(num_constraints <= 0)
    stop("'num_constraints' should be positive")


  # if(gamma <= 0)
  #   stop("'gamma' should be positive")

  if(is.null(constraints) & is.null(label))
    stop("At least one of 'label' and 'constraints' should not be NULL")


  ComputeExtremeDistance <- function(X, a, b, M){
    cat("Computing extreme distance ...")
    if (a < 1 | a > 100)
      stop('a must be between 1 and 100')

    if (b < 1 | b > 100)
      stop('b must be between 1 and 100')

    n <- dim(X)[1]
    num_trials <- min(10000, n*(n-1))
    dists <- c()
    for (i in 1:num_trials) {
      j1 <- ceiling(runif(1) * n)
      j2 <- ceiling(runif(1) * n)
      dists[i] <- (X[j1,]- X[j2,]) %*% M %*% (X[j1,]- X[j2,])
    }


    l <- floor(a/100 * max(dists))
    u <- floor(b/100 * max(dists))

    return(c(l, u))
  }

  GetConstraints <- function(label, num_constraints){
    cat("Getting constraints ...")
    m <- length(label)
    C <- array(0, c(num_constraints, 3))
    k <- 1
    num_diff <- 0
    num_same <- 0
    class <- as.data.frame(table(label))
    while (k <= num_constraints) {
      c1 <- ceiling(runif(1) * dim(class)[1])
      c2 <- ceiling(runif(1) * dim(class)[1])
      all1 <- which(label == class$label[c1])
      all2 <- which(label == class$label[c2])
      i <- all1[ceiling(runif(1) * class$Freq[c1])]
      j <- all2[ceiling(runif(1) * class$Freq[c2])]
      if(c1 == c2 & num_same < num_constraints/2){
        C[k, ] <- c(i, j, 1)
        num_same <- num_same + 1
        k <- k + 1
      }
      else if(c1 != c2 & num_diff < num_constraints/2){
        C[k, ] <- c(i, j, -1)
        num_diff <- num_diff + 1
        k <- k + 1
      }

    }
    return(C)
  }


  ItmlAlg <- function(C, X, params){
    cat("ITML ...")
    tol <- params$thresh
    gamma <- params$gamma
    max_iters <- params$max_iters

    Xdim <- dim(X)

    valid <- array(1, dim(C)[1])
    for (i in 1:dim(C)[1]) {
      i1 <- C[i,1]
      i2 <- C[i,2]
      v <- X[i1,] - X[i2]
      if (sqrt(sum(v^2)) < 10e-10){
        valid[i] <- 0
      }

    }

    C <- C[valid > 0,]

    i <- 1
    iter <- 0
    c <- dim(C)[1]
    lambda <- array(0, c)
    bhat <- C[,4]
    lambdaold <- array(0,c)
    conv <- Inf
    A = diag(1, dim(X)[2])

    while(TRUE){
      i1 <- C[i,1]
      i2 <- C[i,2]
      v <- X[i1,] - X[i2,]
      wtw <- v %*% A %*% v
      if (abs(bhat[i]) < 10e-10) {
        stop('bhat should never be 0!')
      }

      if(Inf == gamma){
        gamma_proj <- 1
      }
      else{
        gamma_proj <- gamma / (gamma + 1)
      }

      if(C[i,3] == 1){
        alpha <- min(lambda[i], gamma_proj * (1/(wtw) - 1/bhat[i]))
        lambda[i] <- lambda[i] - alpha
        beta <- alpha / (1 - alpha * wtw)
        bhat[i] <- solve(1 / bhat[i] + alpha / gamma)
      }
      else{
        alpha <- min(lambda[i], gamma_proj * (1/bhat[i] - 1/(wtw)))
        lambda[i] <- lambda[i] - alpha
        beta <- -alpha / (1 + alpha * wtw)
        bhat[i] <- solve(1 / bhat[i] - alpha / gamma)

      }
      A <- A + beta[1,1] * A %*% v %*% t(v) %*% A

      if(i == c){
        normsum <- sqrt(sum(lambda^2)) + sqrt(sum(lambdaold^2))
        if(normsum == 0){
          break
        }
        else{
          conv <- sum(abs(lambdaold - lambda)) / normsum

          if(conv < tol | iter > max_iters){
            break
          }
        }
        lambdaold <- lambda

      }
      i <- i %% c + 1
      iter <- iter + 1

      if(iter %% c == 0){
        cat('itml iter: ', iter, 'conv = ', conv, '\n')
      }
    }
    return(A)
  }

  drawTSNE <- function(X, label = NULL, legendname = 'cell groups', point_size = 1, labelname = NULL, filename = '0.jpg', colorset = "Set1"){
    if(length(label) == 0){
      label <- array(1, dim(X)[1])
      labelname = c(1)
    }
    p <- ggplot(X, aes(x=X[,1], y=X[,2]))
    p <- p + geom_point(aes(color=factor(label)), size = point_size) + xlab("tSNE1") + ylab("tSNE2")
    p <- p + scale_color_brewer(name=legendname, labels = labelname, type="seq", palette = colorset)
    mytheme <- theme_bw() +
      theme(plot.title=element_text(size=rel(1.5),hjust=0.5),
            axis.title=element_text(size=rel(1)),
            axis.text=element_text(size=rel(1)),
            panel.grid.major=element_line(color="white"),
            panel.grid.minor=element_line(color="white"),
            legend.text = element_text(size = 20),
            legend.title = element_text(size = 25)
      )
    # p + mytheme + guides(colour = guide_legend(override.aes = list(size = 6)))
    print(p + mytheme + guides(colour = guide_legend(override.aes = list(size = 6))))
    #ggsave(filename, dpi = 600)
  }

  A0 <- diag(1, ncol(X))
  extremeDistance <- ComputeExtremeDistance(X, 5, 95, A0)
  print(extremeDistance)
  l <- extremeDistance[1]
  u <- extremeDistance[2]
  gamma <- 10000
  params <- data.frame(thresh, gamma, max_iters)
  if (is.null(constraints)){
    constraints <- GetConstraints(label, num_constraints)
    #save(constraints, file="constraints.Rdata")
  }
  else{
    if (num_constraints > nrow(constraints)){
      constraints <- rbind(constraints, GetConstraints(label, num_constraints - nrow(constraints)))
    }
  }
  constraints <- constraints[1:num_constraints,]
  isSimilar <- u * (1 - constraints[,3]) / 2  + l * (constraints[,3] + 1) / 2
  constraints <- cbind(constraints, isSimilar)
  print(constraints)
  M <- ItmlAlg(constraints, X, params)
  L = chol(M)
  X_new <- X %*% t(L)

  #find key genes
  delta <- array(1, c(dim(L)[2], 1))
  w <- array(1, dim(L)[2])
  for (i in 1:dim(L)[2]) {
    # w[i] <- 2 * t(L[,i]) %*% (L %*% delta) + t(L[,i]) %*% L[,i]
    w[i] <- abs(2 * t(L[,i]) %*% (L %*% delta))
  }
  sortw <- sort(w, index.return = TRUE, decreasing = TRUE)
  sortw$ix <- colnames(X)[sortw$ix]
  #save(sortw, file="sortw.Rdata")

  #draw tSNE plot
  if(draw_tSNE){
    #draw tsne plot
    tsneresult1 <- Rtsne(X, perplexity = 100, pca = TRUE)
    twoD1 <- as.data.frame(tsneresult1$Y)
    drawTSNE(X=twoD1, label = label, legendname='cell groups', labelname = c(1:length(unique(label))), filename="euclidean_metric.jpg")

    tsneresult2 <- Rtsne(X_new, perplexity = 100, pca = TRUE)
    twoD2 <- as.data.frame(tsneresult2$Y)
    drawTSNE(X=twoD2, label = label, legendname='cell groups', labelname = c(1:length(unique(label))), filename="new_metric.jpg")
  }
  res <- list(newData = X_new, newMetric = M, constraints = constraints, sortGenes = sortw)
  return(res)
}
