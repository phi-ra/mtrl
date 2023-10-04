# Price calculation
function_monopoly <- function(p){
  res_ <- -(2*exp(8)*((4*p-5)*exp(4*p)-2*exp(8)))/(exp(4*p)+2*exp(8))^2
  return(res_)
}

res_list <- c()
s_ <- seq(1, 4.5, 0.01)
for (p in s_){
  res_list <- c(res_list, function_monopoly(p))
}

plot(res_list)

print(paste('Monopoloy Price:', s_[which.min(abs(res_list - 0))]))

funcion_duopoly <- function(p){
  a=p
  q=p
  res_ <- -(exp(4*a+8)*(((4*exp(4*a)+4*exp(8))*q-5*exp(4*a)-5*exp(8))*exp(4*q)-exp(4*a+8)))/((exp(4*a)+exp(8))*exp(4*q)+exp(4*a+8))^2
  return(res_)
}

res_list <- c()
s_ <- seq(1.35,1.50, 0.000001)
for (p in s_){
  res_list <- c(res_list, funcion_duopoly(p))
}
print(paste('Duopoly Price:', s_[which.min(abs(res_list - 0))]))


function_five_comp <- function(q){
  a=q
  b=q
  d=q
  c=q
  res_ <- (4*(q-1)*exp(8*(2-q)))/(exp(4*(2-q))+exp(4*(2-d))+exp(4*(2-c))+exp(4*(2-b))+exp(4*(2-a))+1)^2-(4*(q-1)*exp(4*(2-q)))/(exp(4*(2-q))+exp(4*(2-d))+exp(4*(2-c))+exp(4*(2-b))+exp(4*(2-a))+1)+exp(4*(2-q))/(exp(4*(2-q))+exp(4*(2-d))+exp(4*(2-c))+exp(4*(2-b))+exp(4*(2-a))+1)
  return(res_)
}


res_list <- c()
s_ <- seq(1.25,1.35, 0.000001)
for (p in s_){
  res_list <- c(res_list, function_five_comp(p))
}

print(paste('Five Agent Price:', s_[which.min(abs(res_list - 0))]))
