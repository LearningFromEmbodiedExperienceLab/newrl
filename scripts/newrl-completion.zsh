
_newrl() {
    local node=newrl next key
    local -a options
    local -i i
    for (( i = 2; i < CURRENT; i++ )); do
        (( ${+_newrl_commands[$node]} )) || break
        next=${words[i]}
        options=(${=_newrl_commands[$node]})
        (( ${options[(Ie)$next]} )) || return 1
        node+=:$next
    done

    if [[ $PREFIX == -* ]]; then
        options=(${=_newrl_flags[$node]})
        compadd -- $options
    elif (( ${+_newrl_commands[$node]} )); then
        options=(${=_newrl_commands[$node]})
        compadd -- $options
    elif [[ $PREFIX == *=* ]]; then
        key="${node}:${PREFIX%%=*}"
        (( ${+_newrl_values[$key]} )) || return 1
        compset -P '*=' || return 1
        options=(${=_newrl_values[$key]})
        compadd -S '' -- $options
    else
        options=(${=_newrl_keys[$node]})
        if [[ $PREFIX != *.* ]]; then
            options=(${options:#*.*})
        fi
        (( $#options )) || return 1
        compadd -S '=' -- $options
    fi
}

autoload -Uz compinit
(( ${+functions[_complete]} )) || compinit
compdef _newrl newrl
