#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(pracma)
library(ggplot2)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  #Globals
  linStatus <- ""
  nnStatus <- ""
  dimensionality = 2; # Let's stay within x-y dimensions
  
  createData <- reactive({
    # Set seed to get consistent results
    set.seed(1234);
    points = input$points;
    classes = input$classes;
    X = matrix(rep(0), nrow = points*classes, ncol = dimensionality) # data matrix (each row = single example)
    y = matrix(rep(0), nrow = points*classes, ncol = 1) # class labels
    for (j in 1:classes) {
      ix = (points*(j-1)+1):(points*j)
      r = linspace(0.0,1,points) # radius
      t = linspace(j*4,(j+1)*4,points) + randn(1,points) * 0.2 # theta
      X[ix,] = matrix(data = c(r*sin(t), r*cos(t)), ncol = 2)
      y[ix,] = j
    }
    df <- as.data.frame(cbind(X,y))
    colnames(df) <- c("x", "y", "Label")
    df
  })
  
  # Generate a visualisation plot from the generated data.
  output$initial_data <- renderPlot({
    df <- createData()
    points = input$points;
    classes = input$classes;
    p <- ggplot(df, aes(x = x, y = y, color = as.factor(Label))) +
      geom_point() +
      labs(color='Class labels') 
    p
    
  })
  
  # Plot the training error for the linear classifier
  output$linearClassifier <- renderPlot({
    df <- createData()
    cols <- dim(df)[2]
    X = as.matrix(df[,1:(cols-1)])
    y = as.matrix(df[,cols])
    points = input$points;
    classes = input$classes;
    # Hyper paramenters
    reg_lambda = 1e-3
    step_size = 1e-0
    epocs = input$epocs
    # Build one-hot label table
    labels <- y == 1
    for(i in 2:classes) {
      tmp <- y == i
      labels <- cbind(labels, tmp)
    }
    #Let's build the linear classifier model
    #Initialize parameters
    W = 0.01 * randn(dimensionality,classes)
    b = matrix(rep(0),nrow=(points*classes), ncol=classes)
    trainError = data.frame(iter=numeric(),
                          iter_error=numeric(), 
                          stringsAsFactors=FALSE)
    for(i in 0:epocs) {
      scores = X %*% W + b
      # Compute the loss
      num_examples = points * classes
      # get unnormalized probabilities
      exp_scores = exp(scores)
      # normalize them for each example
      probs = exp_scores / rowSums(exp_scores)
      
      correct_logprobs = -log(probs[labels]) #-log(probs[range(num_examples),labels[,"label"]])
      # compute the loss: average cross-entropy loss and regularization
      ln_data_loss = sum(correct_logprobs)/num_examples
      ln_reg_loss = 0.5*reg_lambda*sum(W*W)
      ln_loss = ln_data_loss + ln_reg_loss
      if(i %% 1000 == 0) {
        ln_stat <- paste("Iteration:", i, "loss:", ln_loss ,sep = " ")
        linStatus <- paste(linStatus, ln_stat, sep = " ")
        output$linearClassifierStatus <- renderUI({linStatus})
      }
      #Back-prop
      dscores = probs
      dscores[labels] <- dscores[labels] - 1
      dscores = dscores/num_examples
      dW = t(X) %*% dscores
      db = colSums(dscores)
      dW = dW + reg_lambda*W
      # perform a parameter update
      W = W - step_size * dW
      b = t(t(b)- step_size * db)
      #b = apply(b,1,function(x) {x- step_size * db}) 
      #b - step_size * db
      pre_scores = X %*% W + b
      predicted_class = max.col(pre_scores, 'first')
      iter_err = data.frame(iter = i, iter_error = mean(predicted_class != y))
      trainError = rbind(trainError, iter_err)
    }
    # evaluate training set accuracy
    scores = X %*% W + b
    predicted_class = max.col(scores, 'first') # argmax(scores, axis=1)
    ln_stat <- paste("Iteration:", epocs, "loss:", ln_loss, "Training accuracy:", mean(predicted_class == y), sep = " ")
    linStatus <- paste(linStatus, ln_stat, sep = " ")
    output$linearClassifierStatus <- renderText(linStatus)
    p <- ggplot(trainError, aes(x = iter, y = iter_error)) +
      geom_line() 
    p
    
  })
  nnet <- eventReactive(input$train,{
    df <- createData()
    cols <- dim(df)[2]
    X = as.matrix(df[,1:(cols-1)])
    y = as.matrix(df[,cols])
    points = input$points;
    classes = input$classes;
    epocs = input$epocs;
    hidden_layers = input$layers;
    units_per_layer = input$units;
    #Split the unit string to layers
    hidden_units = unlist(strsplit(units_per_layer, ","))
    #make sure all of the given inputs are numeric
    pass = TRUE
    for(x in hidden_units) {
      units = as.numeric(x)
      output$neuralNetStatus <- renderText(units)
      if(is.na(units)) {
        pass = FALSE
        output$neuralNetStatus <- renderText("Please keep the values for units in each layer between 1 and 1000")
        trainError = NULL
      }
    }
    # All were numbers in given range, now convert to numeric
    hidden_units <- as.numeric(hidden_units)
    # Check that we have amout of neurons for each layer
    if(hidden_layers != length(hidden_units)) {
      pass = FALSE
      output$neuralNetStatus <- renderText("Please give amount of neurons for each hidden layer")
      trainError = NULL
    }
    #If we are good to go
    if(pass) {
      # Hyper paramenters
      reg_lambda = 1e-3
      step_size = 0.1 #1e-0
      
      # Build one-hot label table
      labels <- y == 1
      for(i in 2:classes) {
        tmp <- y == i
        labels <- cbind(labels, tmp)
      }
      #Initialize parameters
      parameters = initialize_parameters(hidden_units, dimensionality, points, classes)
      init_params = parameters;
      trainError = data.frame(iter=numeric(),
                              iter_error=numeric(), 
                              stringsAsFactors=FALSE)
      activations = list()
      activations[[1]] = X
      scores = data.frame(layer=numeric(), score=numeric(), stringsAsFactors = FALSE)
      for(i in 0:epocs) {
        #Compute the forward propagation
        #scores = data.frame(layer=numeric(), score=numeric(), stringsAsFactors = FALSE)
        for(j in 1:length(hidden_units)) {
          activation = relu_activation(parameters[[j]][[1]],parameters[[j]][[2]],activations[[j]])
          #message(class(activation))
          activations[[j+1]] = activation
        }
        #Calculate scores in the output layer
        scores = activations[[length(activations)]] %*% parameters[[length(parameters)]][[1]] + parameters[[length(parameters)]][[2]]
        # Compute the loss
        loss = nnet_loss(scores, num_examples, labels, reg_lambda, parameters[[length(parameters)]][[1]])
        #message(loss)
        if(i %% 1000 == 0) {
          newStatus <- paste("Iteration:",i,"loss:",loss, sep = " ")
          nnStatus <- paste(nnStatus, newStatus, sep = " ")
          output$neuralNetStatus <- renderText({nnStatus})
          message(nnStatus)
          #message(nnStatus)
        }
        
        #Back-prop
        # compute the class probabilities
        exp_scores = exp(scores)
        # normalize them for each example
        probs = exp_scores / rowSums(exp_scores)
        # Gradient on loss
        dscores = probs
        dscores[labels] <- dscores[labels] - 1
        dscores = dscores/num_examples
        # Back-prop through hidden layers to update params
        parameters = nnet_backprop(activations, parameters, dscores, reg_lambda, step_size)        
        #Calculate the predicted scores
        pre_scores = activations[[length(activations)]] %*% parameters[[length(parameters)]][[1]] + parameters[[length(parameters)]][[2]]
        predicted_class = max.col(pre_scores, 'first')
        #message(paste("Iteration:", i, sep = " "))
        iter_err = data.frame(iter = i, iter_error = mean(predicted_class != y))
        trainError = rbind(trainError, iter_err)
      }
      # evaluate training set accuracy
      scores = activations[[length(activations)]] %*% parameters[[length(parameters)]][[1]] + parameters[[length(parameters)]][[2]]
      predicted_class = max.col(scores, 'first') # argmax(scores, axis=1)
      newStatus <- paste("Iteration:", epocs, "loss:", loss, "Training accuracy:", mean(predicted_class == y), sep = " ")
      nnStatus <- paste(nnStatus, newStatus, sep = " ")
      output$neuralNetStatus <- renderText({nnStatus})
    }
    trainError
  })
  
  output$neuralNet <- renderPlot({
    output$neuralNetStatus <- renderText("Training the network can be slow. Especially if there are many layers and many hidden units in each layer. Please hold...")
    trainError = nnet()
    if(!is.null(trainError)) {
      p <- ggplot(trainError, aes(x = iter, y = iter_error)) +
        geom_line() 
      p
    }
  })
  initialize_parameters <- function(hidden_units, dimensionality, points, classes) {
    parameters = list()
    #Initialize parametersrm
    for(i in 1:(length(hidden_units) + 1)) {
      if(i == 1) {
        #Input layer
        W = 0.01 * randn(dimensionality,hidden_units[i])
        b = matrix(rep(0),nrow=(points*classes), ncol=hidden_units[i])
        parameters[[i]] <- list(W,b)
      } else if(i <= length(hidden_units)) {
        # Hidden layers
        W = 0.01 * randn(hidden_units[i-1],hidden_units[i])
        b = matrix(rep(0),nrow=(points*classes), ncol=hidden_units[i])
        parameters[[i]] <- list(W,b)
      } else {
        #Finally the output layer
        W = 0.01 * randn(hidden_units[length(hidden_units)],classes)
        b = matrix(rep(0),nrow=(points*classes), ncol=classes)
        parameters[[i]] <- list(W,b)
      }
    }
    #Return parameters
    parameters
  }  
  relu_activation <- function(W,b,x){
    z = x %*% W + b
    a = z;
    # ReLU activaton, change all sub-zero elements to zero
    a[a<0]<-0; 
    a
  }
  nnet_loss <- function(scores, num_examples, labels, reg_lambda, W) {
    # compute the class probabilities
    exp_scores = exp(scores)
    # normalize them for each example
    probs = exp_scores / rowSums(exp_scores)
    correct_logprobs = -log(probs[labels]) #-log(probs[range(num_examples),labels[,"label"]])
    # compute the loss: average cross-entropy loss and regularization
    data_loss = sum(correct_logprobs)/num_examples
    reg_loss = 0.5*reg_lambda*sum(W*W)
    loss = data_loss + reg_loss
    loss
  }
  nnet_backprop <- function(activations, parameters, dscores, reg_lambda, step_size) {

    # Backprop for output layer
    dW = t(activations[[length(activations)]]) %*% dscores
    db = colSums(dscores)
    dW = dW + reg_lambda*parameters[[length(parameters)]][[1]]
    dhidden = dscores %*% t(parameters[[length(parameters)]][[1]]) #np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[activations[[length(activations)]] <= 0] = 0
    # perform a parameter update
    parameters[[length(parameters)]][[1]] = parameters[[length(parameters)]][[1]] - step_size * dW
    parameters[[length(parameters)]][[2]] = t(t(parameters[[length(parameters)]][[2]])- step_size * db)
    # Calculate back-prob for each layer
    idx <- length(activations)-1
    for(i in idx:1) {
      # backpropate the gradient to the parameters
      dW = t(activations[[i]]) %*% dhidden
      db = colSums(dhidden)
      dW = dW + reg_lambda*parameters[[i]][[1]]
      dhidden = dhidden %*% t(parameters[[i]][[1]]) #np.dot(dscores, W2.T)
      # backprop the ReLU non-linearity
      dhidden[activations[[i]] <= 0] = 0
      # perform a parameter update
      parameters[[i]][[1]] = parameters[[i]][[1]] - step_size * dW
      parameters[[i]][[2]] = t(t(parameters[[i]][[2]])- step_size * db)
    }
    parameters
  }
  
  
})
