#
# This is the user-interface definition of a Shiny web application. You can
# run the application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  tags$head(
    tags$style(HTML("
      @import url('//fonts.googleapis.com/css?family=Lobster|Cabin:400,700');
      
      .container-fluid > h2 {
        font-family: 'Lobster', cursive;
        font-weight: 500;
        line-height: 1.1;
        #color: #48ca3b;
        text-align: center;
      }
      .container-fluid > h3 {
        font-family: 'Lobster', cursive;
        line-height: 1.1;
        #color: #48ca3b;
        text-align: center;
      }
      .container-fluid > h6 {
        font-family: 'Lobster', cursive;
        line-height: 1.1;
        #color: #48ca3b;
        text-align: center;
      }
      .separator {
        margin: 10 0;
        border-bottom: 1px solid grey;
      }

    "))
  ),
  
  # Application title
  titlePanel("Simple Neural Net Demo"),
  h3("Developing Data Products Week4 Course Work"),
  h6("Author: Matti Niemisto"),
  p("The purpose for this data product is to provide quick and easy tool for testing different deep neural net architectures on classification problem. For comparison the results for simple linear classifier as shown."),
  p("TUTORIAL: Select the amount of training examples, classes and iterations with the slider. Also select the amout of hidden layers and define how many neurons you want in each."),
  p("As an activation function for the hidden layers I used ReLU activation and Softmax unit as the final classifier"),
  p("All layers are fully connected. Lamda is 1e-3 and step size is 0.1. Basic L2 reqularization was implemented as well."),
  p("Please keep in mind that setting the amount of neurons really big for each layer makes the computations run really slow."),
  HTML("<p>You can find the source code for both ui.R and server.R in my GitHub repo <a href='' title = ' '>Developing Data Products - Coursera</a></p>"),
  p("Enjoy!"),
  # Sidebar with a slider input for number of bins 
  sidebarLayout(
    sidebarPanel(
      h3("Generate dummy training data"),
      sliderInput("classes",
                 "Number of classes:",
                 min = 1,
                 max = 10,
                 value = 3),
      sliderInput("points",
                  "Number of 'training' examples:",
                  min = 1,
                  max = 300,
                  value = 100),
      tags$div(class="separator"),
      h3("Training parameters"),
      sliderInput("epocs",
                  "Number of Epocs to train the model:",
                  min = 10,
                  max = 10000,
                  value = 1000),
      tags$div(class="separator"),
      h3("Neural Network parameters"),
      sliderInput("layers",
                  "Hidden layers:",
                  min = 1,
                  max = 5,
                  value = 3),
      textInput("units", 
                label = "Comma separated list of number of neurons in each layer",
                value = "", 
                width = NULL, 
                placeholder = NULL),
      actionButton("train", "Train the Model")
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
       h3("Plot the generated fake data"),
       plotOutput(('initial_data')),
       fluidRow(
         column(6,
                h3("Simple linear classifier"),
                #tabletOutput(("linearClassifierStatus")),
                htmlOutput("linearClassifierStatus"),
                plotOutput(('linearClassifier'))
         ),
         column(6,
                h3("Neural network"),
                h6("Click the Train-button to train the network! "),
                textOutput(("neuralNetStatus")),
                plotOutput(('neuralNet'))
                
         )
       )
    )
  )
))
