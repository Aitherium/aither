# Homebrew formula for AitherADK
# Install: brew install aither-adk
# Or from tap: brew tap aitherium/tap && brew install aither-adk

class AitherAdk < Formula
  include Language::Python::Virtualenv

  desc "Agent Development Kit for AitherOS — build AI agent fleets with any LLM"
  homepage "https://aitherium.com"
  url "https://files.pythonhosted.org/packages/source/a/aither-adk/aither_adk-2.42.2.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "Proprietary"

  depends_on "python@3.12"

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/b1/df/48c586a5fe32a0f01324ee087459e112ebb7224f646c0b5023f5e79e9956/httpx-0.28.1.tar.gz"
    sha256 "75e98c5f16b0f35b567856f597f06ff2270a374470a5c2392242528e3e3e42fc"
  end

  resource "pyyaml" do
    url "https://files.pythonhosted.org/packages/05/8e/961c0007c59b8dd7729d542c61a4d537767a59645b82a0b521206e1e25c2/pyyaml-6.0.3.tar.gz"
    sha256 "d76623373421df22fb4cf8817020cbb7ef15c725b9d5e45f17e189bfc384190f"
  end

  resource "fastapi" do
    url "https://files.pythonhosted.org/packages/0d/fb/fd7671137d9fa3df1d93a2f5111eb982709201724b29f211e4beb2d58688/fastapi-0.140.0.tar.gz"
    sha256 "f338951b82fd74ca8f843163aec43ea1a1ce84d515415a50fa98fa25572a5544"
  end

  resource "uvicorn" do
    url "https://files.pythonhosted.org/packages/a2/65/b7c6c443ccc58678c91e1e973bbe2a878591538655d6e1d47f24ba1c51f3/uvicorn-0.51.0.tar.gz"
    sha256 "f6f4b69b657c312f516dd2d268ab9ae6f254b11e4bac504f37b2ab58b24dd0b0"
  end

  def install
    virtualenv_install_with_resources
  end

  def post_install
    # Create config directory
    (var/"aither").mkpath
  end

  test do
    assert_match "AitherADK", shell_output("#{bin}/aither --help")
  end
end
