syntax on
au BufNewFile, BufRead *.py
	set tabstop=2
	set softtabstop=2
	set shiftwidth=2
	set textwidth=79
	set expandtab
	set autoindent
	set fileformat=unix

if has("autocmd")
  au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif
  endif

