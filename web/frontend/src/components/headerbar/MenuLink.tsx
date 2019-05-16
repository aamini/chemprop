import React, { Component } from 'react';

import { Menu, Icon } from 'antd';

const Item = Menu.Item;

export interface MenuLinkProps {
    text: string;
    link: string;
    icon: string;
}

export class MenuLink extends Component<MenuLinkProps> {
    render() {
        return (
            <Item {...this.props} style={{ paddingTop: 8, paddingBottom: 8 }}>
                <a href={this.props.link}>
                    <Icon type={this.props.icon} />
                    {this.props.text}
                </a>
            </Item>
        );
    }
}

export default MenuLink;
