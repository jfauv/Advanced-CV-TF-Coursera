# This function creates a special "matrix" object
# that can cache its inverse
makeCacheMatrix <- function(M = matrix()) {

    M_inv <- NULL
    set <- function(m) {
        M <<- m
        M_inv <<- NULL
    }
    get <- function() {
    	   M
    }
    setInverse <- function(m_inv) {
        M_inv <<- m_inv
    }
    getInverse <- function() {
        M_inv
    }
    list(set = set, get = get,
        setInverse = setInverse,
        getInverse = getInverse)
}


# This function computes the inverse of the special "matrix"
# returned by makeCacheMatrix above.
# If the inverse has already been calculated
# should retrieve the inverse from the cache
cacheSolve <- function(x, ...) {

    M_inv <- x$getInverse()
    if( !is.null(M_inv) ) {
        message("getting cached data")
        return(M_inv)
    }
    M <- x$get()
    M_inv <- solve(M)
    x$setInverse(M_inv)
    return(M_inv)
}