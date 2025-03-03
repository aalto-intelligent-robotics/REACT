#!/bin/sh
NVIM_VERSION=v0.10.3
NODE_VERSION=21

# CLI development tools
# Install FZF
mkdir -p ${HOME}/.bash
mkdir -p ${HOME}/.zsh
mkdir -p ${HOME}/.local/bin

git clone --depth 1 https://github.com/junegunn/fzf.git ${HOME}/.fzf
${HOME}/.fzf/install
cp -f ${HOME}/.fzf/shell/*.bash ${HOME}/.bash
cp -f ${HOME}/.fzf/shell/*.zsh ${HOME}/.zsh
cp -rf ${HOME}/.fzf/bin/* ${HOME}/.local/bin
rm -rf ${HOME}/.fzf

# Node stuffs
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
echo 'export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"'>> ${HOME}/.bashrc
echo '[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm' >> ${HOME}/.bashrc

NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" # This loads nvm
NODE_PATH=$NVM_DIR/v$NODE_VERSION/lib/node_modules
PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

source ${NVM_DIR}/nvm.sh
nvm install ${NODE_VERSION}
nvm alias default ${NODE_VERSION}
nvm use default

# add node and npm to path so the commands are available
# Neovim installation
wget https://github.com/neovim/neovim/releases/download/${NVIM_VERSION}/nvim-linux64.tar.gz
tar xzvf nvim-linux64.tar.gz
mkdir -p ${HOME}/.local/bin
cp ${HOME}/nvim-linux64/bin/nvim ${HOME}/.local/bin
sudo cp -r ${HOME}/nvim-linux64/share/nvim /usr/share
rm -rf ${HOME}/nvim-linux64 nvim-linux64.tar.gz
