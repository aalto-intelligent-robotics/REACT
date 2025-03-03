# Enable Powerlevel10k instant prompt. Should stay close to the top of ${HOME}/.zshrc.
# Initialization code that may require console input (password prompts, [y/n]
# confirmations, etc.) must go above this block; everything else may go below.
# if [[ -r "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh" ]]; then
#   source "${XDG_CACHE_HOME:-$HOME/.cache}/p10k-instant-prompt-${(%):-%n}.zsh"
# fi

# Enable colors and change prompt:
autoload -U colors && colors

# Enable calculator
autoload zcalc

setopt +o nomatch

# # Enable substitution in the prompt.
# setopt prompt_subst

# Config for the prompt. PS1 synonym.
setopt autocd

# History in cache directory:
HISTSIZE=10000
SAVEHIST=10000
HISTFILE=${HOME}/.cache/zsh/history

# Enabling zoxide
# eval "$(zoxide init zsh)"
# additional shell scripts
# source_z() {
#     source ${HOME}/.zshrc.d/zsh-z/zsh-z.plugin.zsh
#     autoload -U compinit
#     zstyle ':completion:*' menu select
#     zmodload zsh/complist
#     compinit
#     _comp_options+=(globdots)
# }

load_nvm() {
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"  # This loads nvm
}
load_nvm_completion() {
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"  # This loads nvm bash_completion
}

# Lazy loading stuffs to speed up start time
source ${HOME}/.zshrc.d/zsh-lazyload/zsh-lazyload.zsh
# lazyload zshz -- "source_z"
lazyload pip pip3 -- 'eval "$(pip completion --zsh)"'
lazyload nvm npm node nvim vim nnn -- 'export NVM_DIR="$HOME/.nvm"; [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"'

# Some nice zsh utils
source ${HOME}/.zshrc.d/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
# source ${HOME}/.zshrc.d/zsh-highlight-config.zsh
source ${HOME}/.zshrc.d/zsh-autosuggestions/zsh-autosuggestions.zsh

# Basic auto/tab complete:
autoload -U compinit
zstyle ':completion:*' menu select
zmodload zsh/complist
compinit
_comp_options+=(globdots)   # Include hidden files.
#
# Load aliases and shortcuts if existent.
source "$HOME/.config/aliasrc"

# More additional shell scripts
source ${HOME}/.zshrc.d/vi-mode.zsh
source ${HOME}/.zshrc.d/conda.zsh
source ${HOME}/.zshrc.d/fzf-funcs.zsh
source ${HOME}/.zshrc.d/ros_config.zsh
source ${HOME}/.zshrc.d/nvim-funcs.zsh
source ${HOME}/.zshrc.d/yazi.zsh
source ${HOME}/.zshrc.d/nnn.zsh
source ${HOME}/.zshrc.d/zshenvs.zsh
source ${HOME}/.zshrc.d/kitty.zsh
# # P10k prompt
source ${HOME}/.zshrc.d/powerlevel10k/powerlevel10k.zsh-theme
source ${HOME}/.zshrc.d/.p10k.zsh
# source ${HOME}/.config/aliasrc
# ## To customize prompt, run `p10k configure` or edit ${HOME}/.p10k.zsh.
# [[ ! -f ${HOME}/.p10k.zsh ]] || source ${HOME}/.p10k.zsh
# # pip autocompletion
#
# #export TERM=xerm-color
#
# source ${HOME}/.zshrc.d/ssh.zsh
# #THIS MUST BE AT THE END OF THE FILE FOR SDKMAN TO WORK!!!
ssdk() {
    export SDKMAN_DIR="$HOME/.sdkman"
    [[ -s "$HOME/.sdkman/bin/sdkman-init.sh" ]] && source "$HOME/.sdkman/bin/sdkman-init.sh"
}
# FZF source
[ -f ${HOME}/.zshrc.d/.fzf.zsh ] && source ${HOME}/.zshrc.d/.fzf.zsh
# Rust source
[ -f "${HOME}/.cargo/env" ] && source "${HOME}/.cargo/env"
eval "$(zoxide init zsh)"
# # Enabling starship prompt
# eval "$(starship init zsh)"
#
# For Pyright to work with ros packages
export PYTHONPATH=$PYTHONPATH:$HOME/hydra_ws/src/hydra_seg_ros/python:$HOME/hydra_ws/src/instance_matching/python:$HOME/hydra_ws/src/instance_matching_ros/python:$HOME/hydra_ws/src/react_ros/python:$HOME/hydra_ws/src/react/python
