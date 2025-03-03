#!/usr/bin/env zsh

vimsnip_cd() {
  cd ~/.config/nvim/snips/
}  

# get_nvim_theme() {
#   export SYS_THEME=$(gsettings get org.gnome.desktop.interface gtk-theme)
#
#   if [[ $SYS_THEME == *"dark"* ]]; then
#     export NVIM_BG='dark'
#   else
#     export NVIM_BG='light'
#   fi
# }
#
# nv() {
#   get_nvim_theme
#   if [[ -z $1 ]]; then
#     nvim +'set bg='$NVIM_BG'' +Alpha 
#   else
#     nvim +'set bg='$NVIM_BG'' $1
#   fi
# }
#
# vim() {
#   nv $1
# }
