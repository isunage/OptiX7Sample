#include "..\include\TestLib\RTScene.h"

auto test::RTScene::GetModel(const std::string& name) const -> const std::shared_ptr<RTModel>&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_Models.at(name);
}

void test::RTScene::SetModel(const std::string& name, const std::shared_ptr<RTModel>& model)
{
    m_Models[name] = model;
}

auto test::RTScene::GetModels() const noexcept -> const ModelMap&
{
    // TODO: return �X�e�[�g�����g�������ɑ}�����܂�
    return m_Models;
}
