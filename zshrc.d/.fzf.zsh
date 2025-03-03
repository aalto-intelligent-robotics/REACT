# Setup fzf
# ---------
if [[ ! "$PATH" == *${HOME}/sys_tools/fzf/bin* ]]; then
    PATH="${PATH:+${PATH}:}${HOME}/sys_tools/fzf/bin"
fi

# Auto-completion
# ---------------
[[ $- == *i* ]] && source "${HOME}/.zsh/completion.zsh" 2> /dev/null

# Key bindings
# ------------
source "${HOME}/.zsh/key-bindings.zsh"
