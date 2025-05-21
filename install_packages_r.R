packages <- c("ggplot2", "dplyr", "tidyr", "data.table")  # list your packages here

install_if_missing <- function(p) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org/")
  }
}

invisible(sapply(packages, install_if_missing))