reset_kitty() {
  kill -SIGUSR1 $(pgrep kitty) 
}
